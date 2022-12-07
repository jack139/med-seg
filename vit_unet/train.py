# coding=utf-8

# python3 -m vit_unet.train

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from .vit_unet import *
from .data import *
from .evaluate import evaluate


batch_size = 32
learning_rate = 1e-4
steps_per_epoch = 1000
epochs = 1
rounds = 30
input_size = (128,128,3)

d_inner_hid=256 # MLP_size
layers=4
n_head=8
d_model=256 # Hidden_size_D

train_path = '../data/train_1h_15m'  
mfile = "vit_unet/vit-unet_%d_%d_%d.weights"%(epochs,batch_size,steps_per_epoch)

if __name__ == '__main__':
    # 新训练
    #model = unet(input_size=input_size, d_inner_hid=d_inner_hid, layers=layers, n_head=n_head, d_model=d_model, lr=learning_rate)
    # 继续训练
    model = unet(input_size=input_size, d_inner_hid=d_inner_hid, layers=layers, n_head=n_head, d_model=d_model, lr=learning_rate,
        pretrained_weights="vit_unet/15m_1h_1y_H256_L4_H8_M256_P4_B48/vit-unet_1_48_1000_r5_0.4501.weights")

    data_gen_args = dict(fill_mode='nearest')
    myGene = trainGenerator(batch_size,train_path,'image','mask',data_gen_args,
        target_size=(128,128),save_to_dir = None)

    model_checkpoint = ModelCheckpoint(mfile, monitor='loss',verbose=1, 
        save_best_only=True, save_weights_only=True)

    for rr in range(rounds):
        model.fit_generator(myGene,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[model_checkpoint])

        print('evaluating ...')
        f1 = evaluate(model, d_inner_hid=d_inner_hid, layers=layers, n_head=n_head, d_model=d_model,test_n=500)

        mfile2 = "vit_unet/vit-unet_%d_%d_%d_r%d_%s.weights"%(epochs,batch_size,steps_per_epoch,rr+1,f1)

        if os.path.exists(mfile):
            os.rename(mfile, mfile2)


else:
    model = unet(input_size=input_size, pretrained_weights=mfile)

    test_path = "data/test"
    testGene = testGenerator(test_path, target_size=input_size[:2])
    file_list = os.listdir(test_path)
    results = model.predict_generator(testGene,len(file_list),verbose=1)
    saveResult("data/results",results,mask_num=5)
