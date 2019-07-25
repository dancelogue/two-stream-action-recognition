import os

# Third Party
import numpy as numpy

# NVIDIA
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin import pytorch


class VideoReaderPipeline(Pipeline):
    def __init__(self, batch_size, sequence_length, num_threads, device_id, files, crop_size, transform=None):
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
        self.transform = transform

        def define_graph(self):
            output = self.input(name='Reader')

            if self.transform:
                output = self.transform(output)
            return output


class DaliLoader():
    def __init__(self, batch_size, file_root, sequence_length, crop_size):
        container_files = [file_root + '/' + f for f in os.listdir(file_root)]
        self.pipeline = VideoReaderPipeline(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_threads=2,
            device_id=0,
            files=container_files,
            crop_size=crop_size
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
