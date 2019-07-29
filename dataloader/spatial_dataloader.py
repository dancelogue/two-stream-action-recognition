# Python
import pickle
from pathlib import Path
import random
import subprocess
import os

# App
try:
    from .split_train_test_video import UCF101_splitter
except Exception: #ImportError
    from split_train_test_video import UCF101_splitter

# Pytorch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Third Party
from PIL import Image
from skimage import io, color, exposure


FNULL = open(os.devnull, 'w')


class spatial_dataset(Dataset):
    def __init__(self, dic, root_dir, mode, transform=None):

        self.keys = dic.keys()
        self.values = dic.values()
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.cache_list = []

    def __len__(self):
        return len(self.keys)

    def ucf_image_path(self, video_name, index, video_path=None):
        if video_name.split('_')[0] == 'HandstandPushups':
            n, g = video_name.split('_', 1)
            name = 'HandStandPushups_' + g
            path = self.root_dir + 'HandstandPushups' + '/frame000'
        else:
            path = self.root_dir + 'v_' + video_name + '/frame000'

        str_index = str(index) if index >= 100 else "0%s" % index if index >= 10 else "00%s" % index
        path = path + str_index + '.jpg'
        
        # if video_path and video_path.exists():
            # print(video_path)

        if Path(path).exists():
            return path

        print('Path does not exist', str(path))
        return False


    def load_ucf_image(self, path):
        # if video_name.split('_')[0] == 'HandstandPushups':
        #     n, g = video_name.split('_', 1)
        #     name = 'HandStandPushups_' + g
        #     path = self.root_dir + 'HandstandPushups' +'/separated_images/v_' + name + '/v_' + name + '_'
        # else:
        #     path = self.root_dir + video_name.split('_')[0] + '/separated_images/v_' + video_name + '/v_' + video_name + '_'

        # img = Image.open(path + str(index) + '.jpg')

        img = Image.open(path)
        transformed_img = self.transform(img)
        img.close()

        # After reading data, do clean up to conserve space
        # os.remove(path)

        return transformed_img

    def __getitem__(self, idx):
        keys_list = list(self.keys)
        values_list = list(self.values)
        
        value = values_list[idx]
        label = int(value.get('label')) - 1
        video_path = value.get('video_path')
        video_name = value.get('video_name')

        if self.mode == 'train':            
            video_name, nb_clips = keys_list[idx].split(' ')

            nb_clips = int(nb_clips)

            clips = []
            nb_3 = int(nb_clips / 3)
            nb_2_3 = int(nb_clips * 2 / 3)

            index_1 = random.randint(1, nb_3)
            index_2 = random.randint(nb_3, nb_2_3)
            index_3 = random.randint(nb_2_3, nb_clips + 1)

            # clips.append(index_1)
            # clips.append(index_2)
            # clips.append(index_3)

            frame_path = '/two-stream-action-recognition/datasets/temp/train-{video_name}-frame-{i1}-{i2}-{i3}_%d.jpg'.format(video_name=video_name, i1=index_1, i2=index_2, i3=index_3)
            sub_p = subprocess.call([
                'ffmpeg',
                '-i',
                str(video_path),
                '-vf',
                "select='eq(n\,{i1})+eq(n\,{i2})+eq(n\,{i3})'".format(i1=index_1, i2=index_2, i3=index_3),
                '-vsync',
                '0',
                frame_path
                ],
                stdout=FNULL,
                stderr=subprocess.STDOUT
            )

            [clips.append(frame_path % (i + 1)) for i in range(3)]

        elif self.mode == 'val':
            video_name, index = keys_list[idx].split(' ')
            index = abs(int(index))
        else:
            raise ValueError('There are only train and val mode')
    
        if self.mode == 'train':
            data = {}
            for i in range(len(clips)):
                key = 'img' + str(i)
                path = clips[i]
                # path = self.ucf_image_path(video_name, index, video_path=video_path)
                # if not path:
                #     continue

                data[key] = self.load_ucf_image(path)

            sample = (data, label)

        elif self.mode == 'val':
            # path = self.ucf_image_path(video_name, index)
            frame_path_val = '/two-stream-action-recognition/datasets/temp/val-{video_name}-frame-{index}_%d.jpg'.format(video_name=video_name, index=index)

            subprocess.call([
                'ffmpeg',
                '-i',
                str(video_path),
                '-vf',
                "select='eq(n\,{index})'".format(index=index),
                '-vsync',
                '0',
                frame_path_val
                ],
                stdout=FNULL,
                stderr=subprocess.STDOUT
            )

            
            data = self.load_ucf_image(frame_path_val % 1)
            sample = (video_name, data, label)
        else:
            raise ValueError('There are only train and val mode')

        return sample


class spatial_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, path, ucf_list, ucf_split):

        self.BATCH_SIZE = BATCH_SIZE
        self.num_workers = num_workers
        self.data_path = path
        self.frame_count = {}

        # split the training and testing videos
        splitter = UCF101_splitter(
            path=ucf_list,
            split=ucf_split,
            dataset_path=path
        )
        self.train_video, self.test_video = splitter.split_video()

    def load_frame_count(self):
        with open('/two-stream-action-recognition/dataloader/dic/frame_count.pickle', 'rb') as file:
            dic_frame = pickle.load(file)
        file.close()

        for line in dic_frame:
            videoname = line.split('_', 1)[1].split('.', 1)[0]
            n, g = videoname.split('_', 1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_' + g
            self.frame_count[videoname] = dic_frame[line]

    def run(self):
        self.load_frame_count()
        self.get_training_dic()
        self.val_sample20()
        train_loader = self.train()
        val_loader = self.validate()

        return train_loader, val_loader, self.test_video

    def get_training_dic(self):
        self.dic_training = {}
        for video in self.train_video:
            nb_frame = self.frame_count[video] - 10 + 1
            key = video + ' ' + str(nb_frame)
            self.dic_training[key] = self.train_video[video]

    def val_sample20(self):
        print('==> sampling testing frames')
        self.dic_testing = {}
        for video in self.test_video:
            nb_frame = self.frame_count[video] - 10 + 1
            interval = int(nb_frame / 19)
            for i in range(19):
                frame = i * interval
                key = video + ' ' + str(frame + 1)
                self.dic_testing[key] = self.test_video[video]

    def train(self):
        training_set = spatial_dataset(
            dic=self.dic_training,
            root_dir=self.data_path,
            mode='train',
            transform=transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        )

        print('==> Training data :', len(training_set), 'frames')
        print(training_set[1][0]['img1'].size())

        train_loader = DataLoader(
            dataset=training_set,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers
        )

        return train_loader

    def validate(self):
        validation_set = spatial_dataset(
            dic=self.dic_testing,
            root_dir=self.data_path,
            mode='val',
            transform=transforms.Compose([
                transforms.Scale([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
        )

        print('==> Validation data :', len(validation_set), 'frames')
        print(validation_set[1][1].size())

        val_loader = DataLoader(
            dataset=validation_set,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.num_workers
        )

        return val_loader


if __name__ == '__main__':
    dataloader = spatial_dataloader(
        BATCH_SIZE=1,
        num_workers=1,
        path='/UCF101/jpegs/jpegs_256/',
        ucf_list='/two-stream-action-recognition/UCF_list/',
        ucf_split='01'
    )
    train_loader, val_loader, test_video = dataloader.run()
    print(len(train_loader))

    for i, (item, label) in enumerate(train_loader):
        print(item['img0'].shape, label)
        # print(i, '--', item.shape)
