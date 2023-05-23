import os
from glob import glob
from multiprocessing import Pool

from PIL import Image

from common import object_names


def augment():
    with Pool() as pool:
        pool.map(augment_object, object_names)


def augment_object(object_name: str):
    in_path = f'./data/{object_name}/train/good'
    out_path = f'{in_path}/augmented'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    paths = glob(f'{in_path}/*.png')
    for path in paths:
        save_path = out_path + '/' + path.split('/')[-1].removesuffix('.png')

        image = Image.open(path)
        image.transpose(Image.FLIP_LEFT_RIGHT).save(save_path + '_vflip.png')
        image.transpose(Image.FLIP_TOP_BOTTOM).save(save_path + '_hflip.png')
        image.transpose(Image.ROTATE_90).save(save_path + '_rot90.png')
        image.transpose(Image.ROTATE_180).save(save_path + '_rot180.png')
        image.transpose(Image.ROTATE_270).save(save_path + '_rot270.png')


if __name__ == '__main__':
    augment()
