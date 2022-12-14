# coding=utf-8

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from keras.callbacks import ModelCheckpoint

from model import unet
from data import trainGenerator, testGenerator, saveResult
from eval import evaluate

batch_size = 2
steps_per_epoch = 1000
epochs = 1
rounds = 10
input_size = (512,512,1)

train_path = '../data/DRIVE2004/training'

if __name__ == '__main__':
    # 新训练
    #model = unet(input_size=input_size)
    # 继续训练
    model = unet(input_size=input_size, pretrained_weights="unet_10_1000.hdf5")

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

    model_checkpoint = ModelCheckpoint("unet_%d_%d.hdf5"%(rounds,steps_per_epoch), 
        monitor='loss',verbose=1, save_best_only=True)

    for rr in range(rounds):
        model.fit_generator(
            myGene,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[model_checkpoint]
        )

        #print('evaluating ...')
        f1 = evaluate(model, target_size=input_size[:2], as_gray=True)
        f1 = f1.tolist()
        print(f"AUC={f1[0]:.4f} F1={f1[1]:.4f} Acc={f1[2]:.4f} Sen={f1[3]:.4f} Spe={f1[4]:.4f} pre={f1[5]:.4f} IOU={f1[6]:.4f}")

else:
    model = unet(input_size=input_size, pretrained_weights="unet_10_1000.hdf5")

    test_path = "../data/DRIVE2004/test/images"
    testGene = testGenerator(test_path, target_size=input_size[:2], as_gray=True)
    file_list = os.listdir(test_path)
    results = model.predict_generator(testGene,len(file_list),verbose=1)
    saveResult("data/results",results)