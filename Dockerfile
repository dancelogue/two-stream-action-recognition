FROM nvidia/cuda:9.0-base

RUN apt-get update && apt-get install -y rsync htop git openssh-server

# Python dependencies
RUN apt-get install python3-pip -y
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install --upgrade pip

#Torch and dependencies:
RUN pip install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
ADD requirements.txt /
RUN pip install -r /requirements.txt
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/9.0 nvidia-dali

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.0"

RUN apt-get install -y ffmpeg

# FFmpeg
# RUN FFMPEG_VERSION=3.4.2 && \
#     cd /tmp && wget https://developer.download.nvidia.com/compute/redist/nvidia-dali/ffmpeg-${FFMPEG_VERSION}.tar.bz2 && \
#     tar xf ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
#     rm ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
#     cd ffmpeg-$FFMPEG_VERSION && \
#     ./configure \
#       --prefix=/usr/local \
#       --disable-static \
#       --disable-all \
#       --disable-autodetect \
#       --disable-iconv \
#       --enable-shared \
#       --enable-avformat \
#       --enable-avcodec \
#       --enable-avfilter \
#       --enable-protocol=file \
#       --enable-demuxer=mov,matroska \
#       --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb && \
#     make -j"$(grep ^processor /proc/cpuinfo | wc -l)" && make install && \
#     cd /tmp && rm -rf ffmpeg-$FFMPEG_VERSION