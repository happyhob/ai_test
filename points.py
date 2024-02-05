import ultralytics
from ultralytics import YOLO
from glob import glob
import numpy as np
import json
import cv2
import torch
from PIL import Image


# IMAGE_DIR = "C:/Users/user/OneDrive - 우송대학교(WOOSONG UNIVERSITY)/바탕 화면/structure/image25.jpg"
IMAGE_DIR = "C:/ai_test/dataset/image/img12.jpg"
MODEL_DIR ="C:/ai_test/best5.pt"

image = Image.open(IMAGE_DIR)

# 이미지의 높이와 너비 가져오기
width, height = image.size

print("Width:", width)
print("Height:", height)
print("--------------------------------------------------------------------")

model = YOLO(MODEL_DIR)


result = model.predict(source=IMAGE_DIR, save=True)
points = result[0].masks.xy

#추출한 좌표를 리스트 형태로 만듬
seg =[]
for point in points:
    if isinstance(point, torch.Tensor):
        point = point.cpu().detach().numpy()
    seg.append(point.tolist())
print(seg)



#리스트를 딕셔너리 형로 만듬
point ={}
for i in range(0,len(seg)-1):
    point["points{}".format(i)]=seg[i]

test = point['points0']


# print(point)
# #좌표리스트를 라벨형식으로 변환
# txt_path ='test.txt'
# with open(txt_path,'w') as txt_outfile:
#     for i in range(0,len(seg)-1):
#         if(i!=0): 
#             txt_outfile.write("\n")
#         txt_outfile.write("0 ")
#         for idx in range(0,len(seg[i])):
#             str ="{} {} ".format(seg[i][idx][0], seg[i][idx][1])
#             txt_outfile.write(str)
        














# #리스트를 딕셔너리 형로 만듬
# point ={}
# for i in range(0,len(seg)-1):
#     point["points{}".format(i)]=seg[i]

# test = point['points0']
# print("----------------------------------------------------")
# print(seg[0][0])
# print("----------------------------------------------------")






# dic1 = {
#     "label":"room",
# }
# for i in range(0,len(seg)-1):
#     dic1["points{}".format(i)]=seg[i]

# # with open('output.json', 'w') as json_file:
# #     json.dump(dic1, json_file)

# print(point)



#print(point)
# for point in points :
#     print(point.cpu().detach().numpy().tolist())



