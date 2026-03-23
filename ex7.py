import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lenna.png 파일을 읽어옵니다
img = cv2.imread('data/Lenna.png')

# 이미지가 제대로 읽혔는지 확인합니다
if img is None:
    print("이미지를 읽을 수 없습니다.")
    exit()

# 그레이스케일로 변환합니다
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 히스토그램을 계산합니다
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# 히스토그램을 플롯합니다
plt.plot(hist)
plt.title('Grayscale Histogram of Lenna.png')
plt.xlabel('Pixel Value (0-255)')
plt.ylabel('Frequency')
plt.show()
