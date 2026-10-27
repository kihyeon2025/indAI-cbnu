import cv2
import numpy as np

# Lenna.png 파일을 읽어옵니다
img = cv2.imread('data/Lenna.png')

# 이미지가 제대로 읽혔는지 확인합니다
if img is None:
    print("이미지를 읽을 수 없습니다.")
    exit()

# BGR 채널을 분리합니다 (OpenCV는 BGR 형식)
b, g, r = cv2.split(img)

# 각 색 성분을 흑백 이미지로 출력합니다
cv2.imshow('Blue Channel', b)
cv2.imshow('Green Channel', g)
cv2.imshow('Red Channel', r)

# 키 입력을 기다립니다
cv2.waitKey(0)

# 모든 창을 닫습니다
cv2.destroyAllWindows()
