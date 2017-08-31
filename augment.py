import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import cv2

def augment_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #convert it to hsv
    h, s, v = cv2.split(hsv)
    correction = random.randint(-80,80)
    v2 = v.astype(int)
    v2 += correction
    v2[v2 > 255] = 255
    v2[v2 < 0] = 0
    v = v2.astype('uint8')

    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image

def flip(image,gt_image):
     if (random.uniform (0, 1) > 0.5):
        image = cv2.flip (image, 1)
        gt_image = cv2.flip (gt_image, 1)
     return image,gt_image 


def augment_pipeline(image,gt_image):
    image=augment_brightness(image)
    image,gt_image=flip(image,gt_image)
    return image,gt_image


def gen_aug_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_aug_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
                image,gt_image = augment_pipeline(image,gt_image)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)


                images.append(image)
                gt_images.append(gt_image)
            #print("about to yield :",len(images)," images")
            yield np.array(images), np.array(gt_images)
    return get_aug_batches_fn


