import cv2
import numpy as np

# Lenna.png 파일을 읽어옵니다
img = cv2.imread('data/Lenna.png')

# 이미지가 제대로 읽혔는지 확인합니다
if img is None:
    print("이미지를 읽을 수 없습니다.")
    exit()

# BGR을 YUV 색상 모델로 변환합니다
yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# YUV 채널을 분리합니다
y, u, v = cv2.split(yuv)

# 각 성분을 흑백 이미지로 출력합니다
cv2.imshow('Y', y)
cv2.imshow('U', u)
cv2.imshow('V', v)

# 키 입력을 기다립니다
cv2.waitKey(0)

# 모든 창을 닫습니다
cv2.destroyAllWindows()
