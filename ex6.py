import cv2
import numpy as np

# candies.png 파일을 읽어옵니다
img = cv2.imread('data/candies.png')

# 이미지가 제대로 읽혔는지 확인합니다
if img is None:
    print("이미지를 읽을 수 없습니다.")
    exit()

# BGR을 HSV 색상 모델로 변환합니다
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 빨간색 범위를 정의합니다 (HSV에서 빨간색은 두 범위로 나뉩니다)
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

lower_red2 = np.array([160, 50, 50])
upper_red2 = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

# 두 마스크를 결합합니다
mask = mask1 | mask2

# 마스크를 적용하여 빨간색 캔디만 추출합니다
result = cv2.bitwise_and(img, img, mask=mask)

# 원본 이미지와 추출된 이미지를 출력합니다
cv2.imshow('Original Image', img)
cv2.imshow('Red Candies Extracted', result)

# 키 입력을 기다립니다
cv2.waitKey(0)

# 모든 창을 닫습니다
cv2.destroyAllWindows()
