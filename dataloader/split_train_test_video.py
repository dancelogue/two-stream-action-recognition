import os, pickle
from pathlib import Path


class UCF101_splitter():
    def __init__(self, path, split, dataset_path):
        self.path = path
        self.split = split
        self.dataset_path = dataset_path

    def get_action_index(self):
        self.action_label = {}
        with open(self.path + 'classInd.txt') as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()

        for line in content:
            label, action = line.split(' ')
            if action not in self.action_label.keys():
                self.action_label[action] = label

    def split_video(self):
        self.get_action_index()
        for path, subdir, files in os.walk(self.path):
            for filename in files:
                if filename.split('.')[0] == 'trainlist' + self.split:
                    train_video = self.file2_dic(self.path + filename)

                if filename.split('.')[0] == 'testlist' + self.split:
                    test_video = self.file2_dic(self.path + filename)

        print(
            '==> (Training video, Validation video):(',
            len(train_video),
            len(test_video),
            ')'
        )

        self.train_video = self.name_HandstandPushups(train_video)
        self.test_video = self.name_HandstandPushups(test_video)

        return self.train_video, self.test_video

    def file2_dic(self, fname):
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        dic = {}

        for line in content:
            video = line.split('/', 1)[1].split(' ', 1)[0]
            key = video.split('_', 1)[1].split('.', 1)[0]

            label = self.action_label[line.split('/')[0]]

            if 'handstandpushups' in video.lower():
                folder_name = 'HandstandPushups'
                video_name = 'HandStandPushups_{}'.format("_".join(key.split('_')[1:]))
            else:
                folder_name = key.split('_')[0]
                video_name = key

            video_path = Path(self.dataset_path) / folder_name / "v_{}.avi".format(video_name)

            if not video_path.exists():
                print('VIDEO DOES NOT EXIST: ', video_path)
                continue

            dic[key] = {
                'label': int(label),
                'video_path': str(video_path),
            }
        return dic

    def name_HandstandPushups(self, dic):
        dic2 = {}
        for video in dic:
            n, g = video.split('_', 1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_' + g
            else:
                videoname = video
            dic2[videoname] = dic[video]
        return dic2


if __name__ == '__main__':

    path = '../UCF_list/'
    split = '01'
    splitter = UCF101_splitter(path=path, split=split)
    train_video, test_video = splitter.split_video()
    print(len(train_video), len(test_video))
