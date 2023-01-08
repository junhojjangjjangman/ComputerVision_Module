import cv2
import numpy as np
import warnings
import Dir
import os

from sklearn.preprocessing import LabelEncoder
from sklearn import svm

warnings.filterwarnings('ignore')

## 1.1 특성 추출 - 추론을 돕기 위해 사용할 특성 선택
def averagecolor(image):
    return np.mean(image, axis=(0, 1))

# 이미지를 메모리로 읽어 보겠습니다.
red_card = cv2.imread(Dir.dir+"[Dataset] Module 21 images/cardred_close.png")
green_card = cv2.imread(Dir.dir+"[Dataset] Module 21 images/cardgreen_close.png")
black_card = cv2.imread(Dir.dir+"[Dataset] Module 21 images/cardblack_close.png")
background = cv2.imread(Dir.dir+"[Dataset] Module 21 images/cardnone.png")

print (averagecolor(red_card))
print (averagecolor(green_card))
print (red_card.shape)
print (green_card.shape)

trainX = []
trainY = []

# 카드들에 대해 반복하고 평균 색상 출력
for (card,label) in zip((red_card,green_card,black_card,background),("red","green","black","none")):
    print((label, averagecolor(card)))
    trainX.append(averagecolor(card))
    trainY.append(label)

print(trainX)
print(np.array(trainX).shape)
print(trainY)
print(np.array(trainY).shape)

## 1.2 K-최근접 이웃(K-Nearest Neighbour, kNN) 알고리즘
new_card = cv2.imread(Dir.dir+"[Dataset] Module 21 images/test/16.png")
new_card_features = averagecolor(new_card)

calculated_distances = []
for card in (trainX):
    calculated_distances.append(np.linalg.norm(new_card_features - card))

print(calculated_distances)
# 우리가 사용한 거리 측정은 "np.linalg.norm ()"
# np.argmin의 역할은 무엇입니까?
print(trainY[np.argmin(calculated_distances)])
print(calculated_distances)
print(np.argmin(calculated_distances))
print(trainY)
print(trainY[np.argmin(calculated_distances)])

# 먼저 새로운 이미지를 메모리로 읽습니다.
new_card = cv2.imread(Dir.dir+"[Dataset] Module 21 images/test/36.png")
new_card_features = averagecolor(new_card)

# 새 이미지의 특성(평균 색상)과 알려진 이미지 특성 사이의 거리를 계산합니다.
calculated_distances = []
for card in (trainX):
    calculated_distances.append(np.linalg.norm(new_card_features-card))

# 다음은 가장 유사한 카드의 결과입니다:
print(trainY[np.argmin(calculated_distances)])

from sklearn.metrics import classification_report
# 테스트 이미지에 대한 진리표입니다. 이미지를 보려면 컴퓨터의 폴더를 여십시오.
realtestY = np.array(["black","black","black","black","black",
                     "red","red","red","red","red",
                     "green","green","green","green","green",
                     "none","none","none","none","none"])
def evaluateaccuracy(filenames,predictedY):
    predictedY = np.array(predictedY)
    if (np.sum(realtestY!=predictedY)>0):
        print ("Wrong Predictions: (filename, labelled, predicted) ")
        print (np.dstack([filenames,realtestY,predictedY]).squeeze()[(realtestY!=predictedY)])
    # 전체 예측의 백분율로 일치하는 (정확한) 예측을 계산합니다.
    return "Correct :"+ str(np.sum(realtestY==predictedY)) + ". Wrong: "+str(np.sum(realtestY!=predictedY)) + ". Correctly Classified: " + str(np.sum(realtestY==predictedY)*100/len(predictedY))+"%"


path = Dir.dir+"[Dataset] Module 21 images/test/"
predictedY = []
filenames = []
for filename in os.listdir(path):
    img = cv2.imread(path + filename)
    img_features = averagecolor(img)
    calculated_distances = []
    for card in (trainX):
        calculated_distances.append(np.linalg.norm(img_features - card))
    prediction = trainY[np.argmin(calculated_distances)]

    print(filename + ": " + prediction)  # 추론을 출력합니다.
    filenames.append(filename)
    predictedY.append(prediction)

# 정확도 평가(sklearn 패키지는 유용한 보고서를 제공합니다)
print()
print(classification_report(realtestY, predictedY))

# 정확도 평가(잘못 분류된 항목의 파일 이름을 출력하기 위한 자체 사용자 정의 메소드)
print()
print(evaluateaccuracy(filenames, predictedY))

trainX2 = []
trainY2 = []

# 이미지 하위 디렉토리 4개 폴더에 있는 훈련 이미지를 반복합니다.
path = Dir.dir+"[Dataset] Module 21 images/"
for label in ('red', 'green', 'black', 'none'):
    print("Loading training images for the label: " + label)

    # 하위 폴더의 모든 이미지를 읽어옵니다.
    for filename in os.listdir(path + label + "/"):
        img = cv2.imread(path + label + "/" + filename)
        img_features = averagecolor(img)
        trainX2.append(img_features)
        trainY2.append(label)

print (len(trainX2))
print (len(trainY2))

path = Dir.dir+"[Dataset] Module 21 images/test/"
filenames = []
predictedY = []
for filename in os.listdir(path):
    img = cv2.imread(path + filename)
    img_features = averagecolor(img)
    calculated_distances = []
    for card in (trainX2):
        calculated_distances.append(np.linalg.norm(img_features - card))
    prediction = trainY2[np.argmin(calculated_distances)]

    print(filename + ": " + prediction)
    filenames.append(filename)
    predictedY.append(prediction)

# 정확도 평가(sklearn 패키지는 유용한 보고서를 제공합니다)
print()
print(classification_report(realtestY, predictedY))

# 정확도 평가
print(evaluateaccuracy(filenames, predictedY))

encoder = LabelEncoder()                         # 레이블을 숫자로 인코딩
encodedtrainY2 = encoder.fit_transform(trainY2)  # 레이블을 숫자로 인코딩

model = svm.SVC(gamma="scale", decision_function_shape='ovr')
model.fit(trainX2, encodedtrainY2)

print (encodedtrainY2)

path = Dir.dir+"[Dataset] Module 21 images/test/"
filenames = []
predictedY = []
for filename in os.listdir(path):
    img = cv2.imread(path + filename)
    img_features = averagecolor(img)
    prediction = model.predict([img_features])[0]

    # 예측을 코드화합니다.
    prediction = encoder.inverse_transform([prediction])[0]

    print(filename + ": " + prediction)
    filenames.append(filename)
    predictedY.append(prediction)

# 정확도 평가(sklearn 패키지는 유용한 보고서를 제공합니다)
print()
print(classification_report(realtestY, predictedY))

# 정확도 평가
print(evaluateaccuracy(filenames, predictedY))

imagenew = cv2.imread("[Dataset] Module 21 images/cardtestagain.png")
imagenew_features = averagecolor(imagenew)
prediction = (model.predict([imagenew_features])[0])

# 숫자에서 레이블로 예측 값을 디코딩
print(encoder.inverse_transform([prediction])[0])

calculated_distances = []
for card in (trainX):
    calculated_distances.append(np.linalg.norm(imagenew_features-card))
print(trainY2[np.argmin(calculated_distances)])

imagenew = cv2.imread("[Dataset] Module 21 images/cardtestagain2.png")
imagenew_features = averagecolor(imagenew)
calculated_distances = []
for card in (trainX2):
    calculated_distances.append(np.linalg.norm(imagenew_features - card))

print("SVM: " + str(encoder.inverse_transform([model.predict([imagenew_features])[0]])[0]))
print("kNN: " + str(trainY2[np.argmin(calculated_distances)]))
