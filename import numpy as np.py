import cv2
import webbrowser

#이미지 전처리
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

#나이 모델 로드
age_net = cv2.dnn.readNetFromCaffe(
	'deploy_age.prototxt',
	'age_net.caffemodel')

#나이 리스트
age_list = ['(0 ~ 2)','(4 ~ 6)','(8 ~ 12)','(15 ~ 20)',
            '(25 ~ 32)','(38 ~ 43)','(48 ~ 53)','(60 ~ 100)']

#얼굴 분류 모델
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#노트북 웹캠 오픈
cap = cv2.VideoCapture(0)

#캠 프레임 크기 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#얼굴 및 나이 추정
while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.05, 5)

    if len(faces):
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            faces_roi = frame[int(y):int(y+h),int(x):int(x+h)].copy()
            blob = cv2.dnn.blobFromImage(faces_roi, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_preds.argmax()

            info = age_list[age]
        
        #텍스트 삽입
        cv2.putText(frame,info,(x,y-15),0, 0.5, (0, 255, 0), 1)

        #이미지 보여주기
        cv2.imshow('result', frame)

        #나이에 따른 화면 전환
        if age >= 65:
            webbrowser.open("new.html")
            exit(0)
        else:
            webbrowser.open("0~64.html")
            exit(0)

cap.release()
cv2.destroyAllWindows()