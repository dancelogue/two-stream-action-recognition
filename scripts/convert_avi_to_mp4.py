import argparse
import subprocess
from pathlib import Path
import time


def convert_avi_to_mp4(src, dest):
    src_path = Path(src)
    dest_path = Path(dest)

    if not src_path.exists():
        raise Exception('Source file path does not exist')

    if not dest_path.exists():
        dest_path.mkdir()

    for folder in src_path.iterdir():
        if folder.is_dir():
            dest_path_folder = dest_path / folder.name

            if not dest_path_folder.exists():
                dest_path_folder.mkdir()

            for video in folder.iterdir():
                video_name = video.stem
                subprocess.call([
                    'ffmpeg',
                    '-i',
                    str(video),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', 'faststart', '-strict', '-2',
                    str(dest_path_folder / "{}.mp4".format(video_name))
                ])

            


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser(description='Convert AVI to MP4, a format DALI understands')

    parser.add_argument('--src', default='', type=str, metavar='PATH', help='Folder containing UCF files', required=True)
    parser.add_argument('--dest', default='', type=str, metavar='PATH', help='Folder to write converted files to', required=True)

    arg = parser.parse_args()

    convert_avi_to_mp4(arg.src, arg.dest)

    print('COMPLETED IN: ', time.time() - start)