import ultralytics
from ultralytics import YOLO
from glob import glob
import numpy as np
import json
import cv2
import torch
import os

MODEL_DIR ="C:/ai_test/best4.pt"


#디렉토리에서 이미지 파일 경로 얻어오기
def get_image(file_direction):
    test_image_list = glob(file_direction)
    print(len(test_image_list))
    test_image_list.sort()
    for i in range(len(test_image_list)):
        print('i = ',i, test_image_list[i])
    return test_image_list


#라벨 포인트 리스트 얻기
def get_list(image_path):
    model = YOLO(MODEL_DIR)

    #박스의 인식률
    result = model.predict(source=image_path, save=True)
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
    return point_list


#포인트 리스트로 yolo txt 파일 만들기
def labal_txt(txt_path, point_list):
    with open(txt_path,'w') as txt_outfile:
        for i in range(0,len(point_list)-1):
            if(i!=0): 
                txt_outfile.write("\n")
            txt_outfile.write("0 ")
            for idx in range(0,len(point_list[i])):
                str ="{} {} ".format(point_list[i][idx][0]/640, point_list[i][idx][1]/640)
                txt_outfile.write(str)

#파일 전체 경로에서 이름만 얻어오기
def find_name(path):
    file_name = os.path.basename(path)
    name =file_name.split('.')[0]   
    return name

PATH = "C:/ai_test/dataset/image2/*.jpg"

#경로를 얻어온다
image_path_list = get_image(PATH)
print(image_path_list)


for i in range(0,len(image_path_list)):
    image_path = image_path_list[i]
    point_list = get_list(image_path)

    # 파일 경로에서 파일 이름만 추출
    name = find_name(image_path)
    output_path ="C:/ai_test/dataset/label/"+name+".txt"
    labal_txt(output_path,point_list)