import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lenna.png 파일을 읽어옵니다
img = cv2.imread('data/Lenna.png')

# 이미지가 제대로 읽혔는지 확인합니다
if img is None:
    print("이미지를 읽을 수 없습니다.")
    exit()

# BGR 채널을 분리합니다 (OpenCV는 BGR 형식)
b, g, r = cv2.split(img)

# 각 채널의 히스토그램을 계산합니다
hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

# 히스토그램을 플롯합니다
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(hist_b, color='blue')
plt.title('Blue Channel Histogram')
plt.xlabel('Pixel Value (0-255)')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.plot(hist_g, color='green')
plt.title('Green Channel Histogram')
plt.xlabel('Pixel Value (0-255)')

plt.subplot(1, 3, 3)
plt.plot(hist_r, color='red')
plt.title('Red Channel Histogram')
plt.xlabel('Pixel Value (0-255)')

plt.tight_layout()
plt.savefig('histogram.png')
plt.show()
