# 1. 미리 학습된 얼굴 자료를 통해 얼굴 인식 시스템 만들기
# 2. 인식한 얼굴에 PutText를 이용하여 텍스트 넣기
# 3. DateTime을 왼쪽 상단에 넣기
import cv2
import numpy as np
import sys
import Dir
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

camera = cv2.VideoCapture(0)
# 얼굴 인식 파일 받아오기(이미 학습된 얼굴 데이터 가지고 오는 것)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


if not camera.isOpened():
    print("Camera open failed!") # 열리지 않았으면 문자열 출력
    sys.exit()
else:
    # 카메라 크기 조절
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)  # 가로
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)  # 세로
    while(True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.flip(frame, 1)  # 좌우 대칭
        if not ret:  # 새로운 프레임을 못받아 왔을 때 braek
            break
        # edge = cv2.Canny(frame, 50, 150)# 정지화면에서 윤곽선을 추출
        # inversed = ~frame  # 반전

        # cv2.imshow('frame', frame)
        # cv2.imshow('inversed', inversed)
        # cv2.imshow('edge', edge)
        # cv2.imshow('GrayScale', gray, cv2.resize(frame, (800, 600)))이건 안되는건가 ?

        cv2.putText(gray, str(datetime.now()), (1,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2) # 데이트 타임 넣기

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # 이미지 피라미드에 사용하는 scalefactor
            # scale 안에 들어가는 이미지의 크기가 1.05씩 증가 즉 scale size는 그대로
            # 이므로 이미지가 1/1.05 씩 줄어서 scale에 맞춰지는 것이다.
            minNeighbors=5,  # 최소 가질 수 있는 이웃으로 3~6사이의 값을 넣어야 detect가 더 잘된다고 한다.
            # Neighbor이 너무 크면 알맞게 detect한 rectangular도 지워버릴 수 있으며,
            # 너무 작으면 얼굴이 아닌 여러개의 rectangular가 생길 수 있다.
            # 만약 이 값이 0이면, scale이 움직일 때마다 얼굴을 검출해 내는 rectangular가 한 얼굴에
            # 중복적으로 발생할 수 있게 된다.
            minSize=(30, 30)  # 검출하려는 이미지의 최소 사이즈로 이 크기보다 작은 object는 무시
            # maxSize도 당연히 있음.
        )
        print("Number of faces detected: " + str(len(faces)))

        if len(faces):
            # 좌표 값과 rectangular의 width height를 받게 된다. x,y값은 rectangular가 시작하는 지점의 좌표
            # 원본 이미지에 얼굴의 위치를 표시하는 작업을 함. for문을 돌리는 이유는 여러 개가 검출 될 수 있기 때문.
            for (x, y, w, h) in faces:
                cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(gray, "ACCESS GRANTED", (x-2, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 100, 100), 1)
            # #다른 부분, 얼굴 안에 들어있는 눈과 입 등을 검출할 때 얼굴 안엣 검출하라는 의미로 이용되는 것
            # roi_gray = gray[y:y+h, x:x+w] #눈,입을 검출할 때 이용
            # roi_color = img[y:y+h, x:x+w] #눈,입등을 표시할 때 이용
        cv2.imshow('GrayScale', gray)

        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()