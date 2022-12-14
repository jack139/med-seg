# coding=utf-8

import os
import time
import numpy as np
import skimage.io as io
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from data import load_img

image_path = '../data/DRIVE2004/test/images'
mask_path  = '../data/DRIVE2004/test/1st_manual_png'
results_path = 'data/results_val'

if not os.path.exists(results_path):
    os.makedirs(results_path)

def load_image(filename, target_size, as_gray):
    img = load_img(os.path.join(image_path,filename), target_size, as_gray)
    filename2, _ = os.path.splitext(filename) # 01_test.tif --> 01
    filename2 = filename2.split("_")[0]
    img2 = load_img(os.path.join(mask_path,f"{filename2}_manual1.png"), target_size, as_gray, is_mask=True)
    return img, img2


def get_metrics(predict, target):
    predict_b = predict.flatten().astype(np.uint8)
    target = target.flatten().astype(np.uint8)

    tp = (predict_b * target).sum()
    tn = ((1 - predict_b) * (1 - target)).sum()
    fp = ((1 - target) * predict_b).sum()
    fn = ((1 - predict_b) * target).sum()
    auc = roc_auc_score(target, predict_b)
    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = tp / (tp + fp)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    iou = tp / (tp + fp + fn)
    f1 = 2 * pre * sen / (pre + sen)
    return {
        "AUC": np.round(auc, 4),
        "F1": np.round(f1, 4),
        "Acc": np.round(acc, 4),
        "Sen": np.round(sen, 4),
        "Spe": np.round(spe, 4),
        "pre": np.round(pre, 4),
        "IOU": np.round(iou, 4),
    }


def evaluate(model, target_size = (128,128), as_gray = True):
    image_list = os.listdir(image_path)
    image_list = sorted(image_list)


    metrics = np.zeros([7])

    for item in tqdm(image_list):
        _, imagename = os.path.split(item) # 01_test.tif
        img_test, img_true = load_image(imagename, target_size, as_gray)


        # 预测结果
        results = model.predict(img_test)

        # 对比预测方向
        img_pred = results[0]
        img_true = img_true[0]

        img = img_pred[:,:,0]
        img[img > 0.5] = 1
        img[img <= 0.5] = 0

        #io.imsave(os.path.join(results_path,"%s_predict.png"%imagename),(img*255.0).astype(np.uint8), check_contrast=False)
        #io.imsave(os.path.join(results_path,"%s_true.png"%imagename),(img_true[:,:,0]*255.0).astype(np.uint8), check_contrast=False)

        m = get_metrics(img, img_true[:,:,0])
        metrics += np.array(list(m.values())).astype(np.float32)

    metrics /= len(image_list)

    return metrics