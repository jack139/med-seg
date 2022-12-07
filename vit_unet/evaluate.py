# coding=utf-8

import os
import time
import shutil
import numpy as np
import skimage.io as io
import skimage.transform as trans
from sklearn import metrics
from tqdm import tqdm

image_path = 'data/val_1h/image'
mask_path  = 'data/val_1h/mask'
results_path = 'data/results_val'

def load_image(filename,target_size = (128,128),as_gray = False):
    img = io.imread(os.path.join(image_path,filename),as_gray = as_gray)
    img = img / 255
    img = trans.resize(img,target_size)
    img = np.reshape(img,img.shape+(1,)) if as_gray else img
    img = np.reshape(img,(1,)+img.shape)

    img2 = io.imread(os.path.join(mask_path,filename),as_gray = as_gray)
    img2 = img2 / 255
    img2 = trans.resize(img2,target_size)
    img2 = np.reshape(img2,img2.shape+(1,)) if as_gray else img2
    img2 = np.reshape(img2,(1,)+img2.shape)

    return img, img2


def evaluate(model, d_inner_hid=128, layers=4, n_head=4, d_model=512, mask_num=5, test_n=500):
    if os.path.exists(results_path):
        shutil.rmtree(results_path)

    os.makedirs(results_path, exist_ok=True)

    image_list = os.listdir(image_path)
    image_list = sorted(image_list)

    xxx=0
    f1_all = [0.]*5
    full_zero = 0

    for item in tqdm(image_list[-test_n:]):

        _, imagename = os.path.split(item)

        img_test, img_true = load_image(imagename)

        last_y = None
        for n in range(5):
            # 重复预测时，添加最后nn列数据
            if last_y is not None:
                nn = last_y.shape[1]
                # 左移一列，最右mask_num清零
                img_test[0] = np.roll(img_test[0], -nn, axis=1)
                img_test[0][:,-mask_num-nn:-mask_num] = last_y
                img_test[0][:,-mask_num:] = 0

                #io.imsave(os.path.join(results_path, "test_%d.png"%n),(img_test[0]*255).astype(np.uint8))

            # 预测结果
            results = model.predict(img_test)

            # 保存中间结果，下次生成图片时使用
            last_col = results[0][:,-mask_num:-mask_num+1]
            last_y = last_col

            #io.imsave(os.path.join(results_path, "last_%d.png"%n),(last_y*255).astype(np.uint8))


        # 对比预测方向
        img_pred = results[0]
        img_true = img_true[0]

        # 右移
        img_pred = np.roll(img_pred, mask_num-1, axis=1)
        img_pred[:,:mask_num-1] = 0. 

        #io.imsave(os.path.join(results_path, "o_"+imagename),(img_pred*255).astype(np.uint8))

        time_span = img_pred.shape[0]

        tmp_f1 = [0.]*5

        for top_x in range(5):
            top_n = top_x + 1

            # 计算 TOP-n 的 F1
            target_true = img_true[:,time_span-mask_num:time_span-mask_num+top_n]
            target_pred = img_pred[:,time_span-mask_num:time_span-mask_num+top_n]

            # 只统计 threshold 以上的点
            threshold = 0.05
            target_true[target_true>threshold] = 1
            target_pred[target_pred>threshold] = 1

            target_true[target_true<=threshold] = 0
            target_pred[target_pred<=threshold] = 0

            # 展平，计算 F1
            target_true = target_true.reshape([1, time_span*top_n*3])
            target_pred = target_pred.reshape([1, time_span*top_n*3])

            f1 = metrics.f1_score(target_true[0], target_pred[0])

            f1_all[top_x] += f1
            tmp_f1[top_x] = f1

        #print("F1= %s\t%s"%(' '.join('%.4f'%x for x in tmp_f1), imagename))

        if sum(tmp_f1)<0.0001:
            full_zero += 1

        io.imsave(os.path.join(results_path, imagename),(img_pred*255).astype(np.uint8))

        xxx += 1
        #if xxx==1:
        #    break

    f1s = ['%.4f'%(f1_all[top_n]/xxx) for top_n in range(5)]
    print('average F1=', ' '.join(f1s),
        ' full_zero=', full_zero, '%.4f'%(full_zero/xxx) )

    return f1s[0]