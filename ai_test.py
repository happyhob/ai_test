import ultralytics
from ultralytics import YOLO
from glob import glob
import numpy as np
import json
import cv2
import torch

# IMAGE_DIR = "C:/Users/user/OneDrive - 우송대학교(WOOSONG UNIVERSITY)/바탕 화면/structure/image25.jpg"
IMAGE_DIR = "test.jpg"
MODEL_DIR ="C:/ai_test/best.pt"

model = YOLO(MODEL_DIR)

#박스의 인식률
result = model.predict(source=IMAGE_DIR, save=True)
boxes = result[0].boxes
points = result[0].masks.xy

seg =[]
for point in points:
    if isinstance(point, torch.Tensor):
        point = point.cpu().detach().numpy()
    seg.append(point.tolist())

point_list =[]
for i in range(len(boxes)-1): 
    if(boxes[i].conf.item()>=0.900):
        print(i)
        print(seg[i])
        point_list.append(seg[i])


txt_path ='./test.txt'
with open(txt_path,'w') as txt_outfile:
    for i in range(0,len(point_list)-1):
        if(i!=0): 
            txt_outfile.write("\n")
        txt_outfile.write("0 ")
        for idx in range(0,len(point_list[i])):
            str ="{} {} ".format(point_list[i][idx][0], point_list[i][idx][1])
            txt_outfile.write(str)




# for box in boxes :
#     print(box.xyxy.cpu().detach().numpy().tolist())
    # print(box.conf.cpu().detach().numpy().tolist())
    # print(box.cls.cpu().detach().numpy().tolist())

#print(xy)