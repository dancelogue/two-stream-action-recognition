import os

# Third Party
import numpy as numpy

# NVIDIA
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin import pytorch

class VideoReaderPipeline(Pipeline):
    def __init__(self, batch_size, sequence_length, num_threads, device_id, files, crop_size):
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

        def define_graph(self):
            output = self.input(name='Reader')
            return outputv 