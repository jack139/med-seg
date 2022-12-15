# coding=utf-8

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from keras.callbacks import ModelCheckpoint, Callback

from model import unet
from data import trainGenerator, testGenerator, saveResult
from eval import evaluate

lr = 1e-4
batch_size = 2
steps_per_epoch = 1000
epochs = 30
input_size = (512,512,1)

train_path = '../data/DRIVE2004/training'


# 新训练
model = unet(input_size=input_size, lr=lr)
# 继续训练
#model = unet(input_size=input_size, lr=lr, pretrained_weights="unet_e16_iou_0.62700.weights")


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_iou = 0

    def on_epoch_end(self, epoch, logs=None):
        f1 = evaluate(model, target_size=input_size[:2], as_gray=True)
        f1 = f1.tolist()

        # 保存最优
        if f1[6] >= self.best_val_iou:
            self.best_val_iou = f1[6]
            model.save_weights(f'unet_e{epoch:02d}_iou_{f1[6]:.5f}.weights')

        print(f"AUC: {f1[0]:.4f} F1: {f1[1]:.4f} Acc: {f1[2]:.4f} " + \
            f"IoU: {f1[6]:.4f} best IoU: {self.best_val_iou:.4f}")


if __name__ == '__main__':

    data_gen_args = dict(
                        rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
    myGene = trainGenerator(batch_size,train_path,'images','1st_manual_png', data_gen_args,
        target_size=input_size[:2], save_to_dir = None)

    evaluator = Evaluator()

    model.fit_generator(
        myGene,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:
    test_path = "../data/DRIVE2004/test/images"
    testGene = testGenerator(test_path, target_size=input_size[:2], as_gray=True)
    file_list = os.listdir(test_path)
    results = model.predict_generator(testGene,len(file_list),verbose=1)
    saveResult("data/results",results)