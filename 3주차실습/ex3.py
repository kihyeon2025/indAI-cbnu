import cv2

# 웹캠을 읽어옵니다 (0은 기본 웹캠)
cap = cv2.VideoCapture(0)

# 웹캠이 열렸는지 확인합니다
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 웹캠을 프레임별로 읽어 출력합니다
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    # 이부분에 딥러닝 및 프레임 전처리 코드 추가

    # 프레임을 화면에 출력합니다
    cv2.imshow('Webcam', frame)
    
    # 'q' 키를 누르면 종료합니다
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스를 해제합니다
cap.release()
cv2.destroyAllWindows()
