
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc
from scipy.ndimage import rotate
from scipy.stats import bernoulli
# the utils is based on Upul's github https://github.com/upul/Behavioral-Cloning
# Some useful constants
file_path = '/home/szyh/wangmin/CarND/term-1/data/jungle/driving_log.csv'
#this is the left and right camera's steering coefficent
correction = 0.229


def crop(image, up_ratio, down_ratio):
    assert 0 <= up_ratio < 0.5, 'top_percent should be between 0.0 and 0.5'
    assert 0 <= down_ratio < 0.5, 'top_percent should be between 0.0 and 0.5'

    top = int(np.ceil(image.shape[0] * up_ratio))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * down_ratio))

    return image[top:bottom, :]


def resize(image, dims):
    return scipy.misc.imresize(image, dims)


def random_flip(image, steering_angle, flip_prob=0.5):
    is_flip = bernoulli.rvs(flip_prob)
    if is_flip:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle


def random_gamma(image):
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def random_shear(image, steering_angle, shear_range=200):
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering

    return image, steering_angle


# generate new image based on the raw image
def generate_new_image(image, steering_angle, up_crop_ratio=0.35, down_crop_ratio=0.1,
                       resize_dim=(64, 64), shear_prob=0.9):
    """
    :param image:
    :param steering_angle:
    :param up_crop_ratio:
    :param down_crop_ratio:
    :param resize_dim:
    :param shear_prob:
    :param shear_range:
    :return:
    """
    is_shear = bernoulli.rvs(shear_prob)
    if is_shear == 1:
        image, steering_angle = random_shear(image, steering_angle)

    image = crop(image, up_crop_ratio, down_crop_ratio)

    image, steering_angle = random_flip(image, steering_angle)

    image = random_gamma(image)

    image = resize(image, resize_dim)

    return image, steering_angle


def get_next_image_files(batch_size=64):
    """
    randomly pick up left , right or center image
    :param batch_size:
        Size of the image batch
    :return:
        An list of selected (image files names, respective steering angles)
    """
    data = pd.read_csv(file_path)
    num_of_img = len(data)
    indxs = np.random.randint(0, num_of_img, batch_size)

    image_files_and_angles = []
    for index in indxs:
        rnd_image = np.random.randint(0, 3)
        if rnd_image == 0:
            img = data.iloc[index]['left'].strip()
            angle = data.iloc[index]['steering'] + correction
            image_files_and_angles.append((img, angle))

        elif rnd_image == 1:
            img = data.iloc[index]['center'].strip()
            angle = data.iloc[index]['steering']
            image_files_and_angles.append((img, angle))
        else:
            img = data.iloc[index]['right'].strip()
            angle = data.iloc[index]['steering'] - correction
            image_files_and_angles.append((img, angle))

    return image_files_and_angles


def generate_next_batch(batch_size=64):
    """
    This generator yields the next training batch
    :param batch_size:
        Number of training images in a single batch
    :return:
        A tuple of features and steering angles as two numpy arrays
    """
    while True:
        X_batch = []
        y_batch = []
        images = get_next_image_files(batch_size)
        for img_file, angle in images:
            raw_image = plt.imread(img_file)

            raw_angle = angle
            new_image, new_angle = generate_new_image(raw_image, raw_angle)
            X_batch.append(new_image)
            y_batch.append(new_angle)

        assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'

        yield np.array(X_batch), np.array(y_batch)
