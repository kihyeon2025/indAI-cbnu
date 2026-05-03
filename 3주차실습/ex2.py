import cv2

# test_video.mp4 파일을 읽어옵니다
cap = cv2.VideoCapture('data/test_video.mp4')

# 비디오가 열렸는지 확인합니다
if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
    exit()

# 비디오를 프레임별로 읽어 출력합니다
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 프레임을 화면에 출력합니다
    cv2.imshow('Test Video', frame)
    
    # 'q' 키를 누르면 종료합니다
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 리소스를 해제합니다
cap.release()
cv2.destroyAllWindows()
