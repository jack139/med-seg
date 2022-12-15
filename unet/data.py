# coding=utf-8

from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage.color import rgb2gray
from copy import deepcopy

threshold = 0.5

def adjustData(img, mask):
    if np.max(img) > 1:
        img = img / 255

    if np.max(mask) > 1:
        mask = mask / 255

    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0

    return (img, mask)


def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask)
        yield (img,mask)


def load_img(path, target_size, as_gray, is_mask=False):
    #img = io.imread(path,as_gray = as_gray) # 对tiff不能直接读，需要转换
    img = io.imread(path)
    if as_gray:
        img = rgb2gray(img)
    img = trans.resize(img, target_size) # 调整尺寸

    img, mask = adjustData(img, deepcopy(img))

    if is_mask:
        img = mask

    img = np.reshape(img, img.shape+(1,))
    img = np.reshape(img, (1,)+img.shape)
    return img


def testGenerator(test_path, target_size = (256,256), as_gray = True):
    file_list = os.listdir(test_path)
    file_list = sorted(file_list)
    for i in file_list:
        img = load_img(os.path.join(test_path,i), target_size, as_gray)
        yield img


def saveResult(save_path,npyfile):
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        img[img > threshold] = 1
        img[img <= threshold] = 0
        io.imsave(os.path.join(save_path,"predict_%d.png"%i),(img*255.0).astype(np.uint8), check_contrast=False)
