# Python
import zipfile
import os
from pathlib import Path
import math

# Third Party
import requests
from tqdm import tqdm


def modified_print(msg, type=None):
    print('############### %s ###############' % msg)


def download_ucf():
    url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"

    root = Path.cwd() / 'datasets' / 'ucf101'

    if not root.exists():
        root.mkdir()

    file_name = str(root / 'UCF101.rar')
    download(url, file_name, unzip_path=None, msg='DOWNLOADING UCF')


def download_models():
    urls = [
        {},
    ]

def download(url, file_name, unzip_path=None, msg=None):
    # https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
    resp = requests.get(url, stream=True)
    total_size = int(resp.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0

    modified_print('DOWNLOADING %s DATA' % msg)

    with open(file_name, 'wb') as f:
        for data in tqdm(resp.iter_content(block_size), total=math.ceil(total_size // block_size), unit='KB', unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)

        if total_size != 0 and wrote != total_size:
            modified_print("ERROR, something went wrong", type='ERROR')

    if unzip_path:
        modified_print('UNZIPPING %s DATA' % msg)

        unzip_file(file_name, unzip_path)

        modified_print('COMPLETE UNZIPPING DATA')


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def unzip_file(file_name, unzip_path, remove=False):
    zip_ref = zipfile.ZipFile(file_name, 'r')
    zip_ref.extractall(unzip_path)
    zip_ref.close()

    if remove:
        os.remove(file_name)


if __name__ == '__main__':
    download_ucf()
