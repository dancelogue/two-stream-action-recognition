import os
import argparse

# Third Party
import numpy as numpy
import torchvision.transforms as transforms

# NVIDIA
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin import pytorch


class VideoReaderPipeline(Pipeline):
    def __init__(self, batch_size, sequence_length, num_threads, device_id, files, crop_size, transforms=None):
        super(VideoReaderPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.reader = ops.VideoReader(
            device='gpu',
            filenames=files,
            sequence_length=sequence_length,
            normalized=False,
            random_shuffle=True,
            image_type=types.RGB,
            dtype=types.UINT8,
            initial_fill=16
        )

        self.crop = ops.Crop(device="gpu", crop=crop_size, output_dtype=types.FLOAT)
        self.flip = ops.Flip(device="gpu", horizontal=1, vertical=0)
        self.normalize = ops.NormalizePermute(
            device="gpu",
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            width=224,
            height=224
        )

    def define_graph(self):
        output = self.reader(name='Reader')
        output = self.crop(output)
        output = self.flip(output)
        output = self.normalize(output)
        return output


class DaliLoader():
    def __init__(self, batch_size, file_root, sequence_length, crop_size, transforms=None):
        container_files = [file_root + '/' + f for f in os.listdir(file_root)]

        self.pipeline = VideoReaderPipeline(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_threads=2,
            device_id=0,
            files=container_files,
            crop_size=crop_size,
            transforms=transforms
        )
        self.pipeline.build()
        self.epoch_size = self.pipeline.epoch_size('Reader')
        self.dali_iterator = pytorch.DALIGenericIterator(
            self.pipeline,
            ["data"],
            self.epoch_size,
            auto_reset=True
        )

    def __len__(self):
        return int(self.epoch_size)

    def __iter__(self):
        return self.dali_iterator.__iter__()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing the Dali Loader')

    parser.add_argument('--batch-size', default=5, type=int, metavar='N', help='the batch size')
    parser.add_argument('--root-path', default='', type=str, metavar='PATH', help='file root', required=True)
    parser.add_argument('--sequence-length', default=1, type=int, metavar='N', help='the sequence length')
    parser.add_argument('--crop-size', default=224, type=int, metavar='N', help='the crop size')

    arg = parser.parse_args()

    batch_size = arg.batch_size
    file_root = arg.root_path
    sequence_length = arg.sequence_length
    crop_size = arg.crop_size

    loader = DaliLoader(
        batch_size,
        file_root,
        sequence_length,
        crop_size
    )

    for i, inputs in enumerate(loader):
        inputs = inputs[0]["data"]
        print(i, ' -- ', inputs.shape)
