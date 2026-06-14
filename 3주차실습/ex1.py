import cv2

# Lenna.png 파일을 읽어옵니다
img = cv2.imread('data/Lenna.png', cv2.IMREAD_GRAYSCALE)

# 이미지를 화면에 출력합니다
cv2.imshow('Lenna Image', img)

# 키 입력을 기다립니다
cv2.waitKey(0)

# 모든 창을 닫습니다
cv2.destroyAllWindows()
