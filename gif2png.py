# coding=utf-8

import os
from glob import glob
import cv2
from PIL import Image

gif_path = "data/DRIVE2004/training/1st_manual"
output_dir = "data/DRIVE2004/training/1st_manual_png"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_list = glob(f"{gif_path}/*.gif")

for i in file_list:
    _, fn = os.path.split(i)
    fn, _ = os.path.splitext(fn)

    gif = cv2.VideoCapture(i)
    ret,frame = gif.read() # ret=True if it finds a frame else False. Since your gif contains only one frame, the next read() will give you ret=False
    img = Image.fromarray(frame)
    img = img.convert('RGB')
    img.save(os.path.join(output_dir, f'{fn}.png'))
