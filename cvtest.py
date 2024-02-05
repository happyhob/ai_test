from ultralytics import YOLO
from glob import glob
import numpy as np
import json
import cv2
import torch
from PIL import Image
import os




def get_image(file_direction):
    test_image_list = glob(file_direction)
    print(len(test_image_list))
    test_image_list.sort()
    for i in range(len(test_image_list)):
        print('i = ',i, test_image_list[i])
    return test_image_list

PATH = "C:/Users/user/OneDrive - 우송대학교(WOOSONG UNIVERSITY)/바탕 화면/structure/*.jpg"
PATH2 = "C:/Users/user/OneDrive - 우송대학교(WOOSONG UNIVERSITY)/바탕 화면/structure\image1.jpg"

file_name = os.path.basename(PATH2)
name =file_name.split('.')[0]

print("파일 이름:"+name)

image_list =get_image(PATH)

print('----------------------------------------------------------------')
print(image_list[0])
print(image_list[1])
print(image_list[2])

IMAGE_DIR=image_list[0]
MODEL_DIR ="C:/ai_test/best.pt"


model = YOLO(MODEL_DIR)


result = model.predict(source=IMAGE_DIR, save=True)
mask = result[0].masks
xy_data = mask.xy

print(xy_data)