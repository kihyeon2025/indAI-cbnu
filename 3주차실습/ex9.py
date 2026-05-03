import cv2
import numpy as np

# desert.JPG 파일을 읽어옵니다
img = cv2.imread('data/desert.JPG')

# 이미지가 제대로 읽혔는지 확인합니다
if img is None:
    print("이미지를 읽을 수 없습니다.")
    exit()

# 그레이스케일로 변환합니다
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Canny 엣지 검출을 수행합니다
edges = cv2.Canny(gray, 100, 200)

# 원본 이미지와 엣지 이미지를 출력합니다
cv2.imshow('Original Image', img)
cv2.imshow('Edges', edges)

# 키 입력을 기다립니다
cv2.waitKey(0)

# 모든 창을 닫습니다
cv2.destroyAllWindows()
