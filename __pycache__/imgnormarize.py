from PIL import Image
from glob import glob
import os


def resize_image(input_path, output_path, target_size=(640, 640)):
    try:
        # 이미지 열기
        img = Image.open(input_path)

        # 이미지를 원하는 크기로 조정
        img_resized = img.resize(target_size)

        # 조정된 이미지를 저장
        img_resized.save(output_path)

        print(f"이미지가 성공적으로 {target_size} 크기로 조정되었습니다.")
    except Exception as e:
        print(input_path)
        print(f"오류: {e}")


#디렉토리에서 이미지 파일 경로 얻어오기
def get_image(file_direction):
    test_image_list = glob(file_direction)
    print(len(test_image_list))
    test_image_list.sort()
    for i in range(len(test_image_list)):
        print('i = ',i, test_image_list[i])
    return test_image_list


#파일 전체 경로에서 이름만 얻어오기
def find_name(path):
    file_name = os.path.basename(path)
    name =file_name.split('.')[0]   
    return name




IMG ="C:/ai_test/dataset/image/*.jpg"
OUTPUT = "C:/ai_test/dataset/image2/"
img_list  = get_image(IMG)


for i in range(0,len(img_list)):
    name =find_name(img_list[i])
    output_path = OUTPUT+name+".jpg"
    resize_image(img_list[i],output_path)

