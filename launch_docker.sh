#!/bin/bash
sudo nvidia-docker build -t dancelogue:two-stream-action-recognition .
sudo nvidia-docker run --rm -ti --volume=$(pwd):/two-stream-action-recognition:rw --workdir=/two-stream-action-recognition --ipc=host dancelogue:two-stream-action-recognition /bin/bash


# sudo nvidia-docker run --rm -ti --volume="/media/gitumarkk/Extreme SSD/Dancelogue/DATASETS/UCF101":/UCF101:rw --volume=$(pwd):/two-stream-action-recognition:rw --workdir=/two-stream-action-recognition --ipc=host dancelogue:two-stream-action-recognition /bin/bash

