import cv2
import numpy as np
import sys
import Dir
import pandas as pd
import matplotlib.pyplot as plt

print("You have successfully installed OpenCV version "+cv2.__version__)
print("Your version of Python is " + sys.version)

img = cv2.imread(Dir.dir+"[Dataset] Module 20 images/image001.png",-1)
img2 = cv2.imread(Dir.dir+"[Dataset] Module 20 images/image001.png",0)
img3 = cv2.imread(Dir.dir+"[Dataset] Module 20 images/image001.png",1)
# imread(경로, 옵션) 옵션 값 -1: 원본, 0: gray color, 1: BGR색으로 읽기
# cv2.IMREAD_UNCHANGED or -1 : image 파일 변형 없이 원본 읽기
# cv2.IMREAD_COLOR or 1 : BGR 색으로 읽기
# cv2.IMREAD_GRAYSCALE or 0 : 회색으로 이미지 출력하기
# cv2.IMREAD_REDUCED_GRAYSCALE_2 : 회색 출력, 사이즈 반으로 줄이기
# cv2.IMREAD_REDUCED_COLOR_2 : BGR 출력, 사이즈 반으로 줄이기
# cv2.IMREAD_REDUCED_GRAYSCALE_4 : 회색 출력, 사이즈 1/4로 줄이기
# cv2.IMREAD_REDUCED_COLOR_4 : BGR 출력, 사이즈 1/4로 줄이기
# cv2.IMREAD_ANYDEPTH : 8/16/32비트 변경
# cv2.IMREAD_ANYCOLOR : 어떤 색으로든 출력 가능
# cv2.IMREAD_LOAD_GDAL : gdal 드라이브로 이미지 읽기
# cv2.IMREAD_IGNORE_ORIENTATION : EXIF flag에 따라 이미지 회전 하지 않음
if img is None:
    print('Image load failed')
    sys.exit()
cv2.imshow("Image", img)
cv2.imshow("Image2", img2)
cv2.imshow("Image3", img3)

# cv2.moveWindow("Image", 500, 500)
cv2.waitKey(0)# waitKey를 해주지 않으면 자동으로 꺼짐

cv2.destroyAllWindows() #함수는 열린 모든 창을 닫습니다.
# cv2.destroyWindow(winname) 함수를 호출하면 winname에 해당하는 창을 닫습니다.
# cv2.resizeWindow(winname, width, hegith) 함수는 winname 창의 크기를 (width, height) 크기로 변경해줍니다.

print(img[:,:,:].shape)
print(img2[:,:].shape)
img_1 = cv2.cvtColor(img,0)
plt.subplot(2,1,1);plt.imshow(img)
plt.subplot(2,1,2);plt.imshow(img_1)
plt.show()

# 1. 카메라 열기 - cv2.VideoCapture
# 함수 원형 cv2.VideoCapture(index, apiPreference=None) -> retval
# index : camera_id + domain_offset_id 시스템 기본 카메라를 기본 방법으로 열려면 index에 0을 전달합니다.
# 장치관리자에 등록되어 있는 카메라 순서대로 인덱스가 설정되어 있습니다.
# apiPreference : 선호하는 카메라 처리 방법을 지정합니다.
# retval : cv2.VideoCapture 객체를 반환합니다.  retval : 성공하면 True, 실패하면 False를 반환합니다.

# 2. 동영상 열기 - cv2.VideoCapture
# 함수 원형 : cv2.VideoCapture(filename, apiPreference=None) -> retval
#  OpenCV를 이용해서 동영상 여는 방법은 카메라 여는 방법과 동일합니다.
#  차이점은 cv2.VideoCapture() 안에 인덱스 대신에 파일명을 넣어주면 됩니다.
# filename : 비디오 파일 이름, 정지 영상 시퀀스, 비디오 스트림 URL 등, ex) 'video.avi', 'img_%02d.jpg', 'protocol://host:port/script?params|auth'
# apiPreference : 선호하는 카메라 처리 방법을 지정합니다.
# retval : cv2.VideoCapture 객체를 반환합니다. retval : 성공하면 True, 실패하면 False를 반환합니다.

# cv2. 속성 값 참조 할 때
# 함수 원형 : cv2.VideoCapture.get(propId) -> retval

# cv2 속성 값 변경
# 함수원형 : cv2.VideoCapture.set(propId, value) -> retval
# camera.set(cv2.CAP_PROP_FRAME_WIDTH or 3, 320)\
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT or 4, 240)
# 이처럼 640, 480 크기의 영상을 320, 240 크기로 변환하여 출력할 수 있습니다.

# 카메라에서 사진 찍기
# 원형 : cv2.imwrite(filename, img, [parameters])
camera = cv2.VideoCapture(0)
#camera.set(3, 250)
#camera.set(4, 250)

if not camera.isOpened():
    print("Camera open failed!") # 열리지 않았으면 문자열 출력
    sys.exit()
else:
    while(True):
        ret, frame = camera.read()
        # frame = cv2.flip(frame, 1)  # 좌우 대칭
        if not ret:  # 새로운 프레임을 못받아 왔을 때 braek
            break

        edge = cv2.Canny(frame, 50, 150)# 정지화면에서 윤곽선을 추출

        inversed = ~frame  # 반전

        cv2.imshow('frame', frame)
        cv2.imshow('inversed', inversed)
        cv2.imshow('edge', edge)

        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

img_1 = cv2.imread(Dir.dir+"[Dataset] Module 20 images/image001.png")
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
plt.imshow(img_1)# openCV: BGR순 matplotlib: RGB순
plt.title('Image1')
plt.axis('off')
plt.show()
plt.figure(figsize=(10, 2))
plt.subplot(141)
plt.imshow(img_1[:, :, :])
plt.axis("off")
plt.title("RGB Image")
plt.subplot(142)
plt.imshow(img_1[:, :, 0], cmap=plt.cm.bone)
plt.axis("off")
plt.title("R Channel")
plt.subplot(143)
plt.imshow(img_1[:, :, 1], cmap=plt.cm.bone)
plt.axis("off")
plt.title("G Channel")
plt.subplot(144)
plt.imshow(img_1[:, :, 2], cmap=plt.cm.bone)
plt.axis("off")
plt.title("B Channel")
plt.show()

print(img[0, 0])
print(img[0, 799])
h,w,c = img.shape
print("W:", w, "H:", h)
print(img[(w-1)//2, (h-1)//2])
