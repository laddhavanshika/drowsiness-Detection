import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#training Data
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(gray) 
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        #print(roi_gray)
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey),(ex+ew, ey+eh),(0,255,0),2)
    cv2.imshow('img',img)
    k = cv2.waitKey(1) & 0Xff
    if k ==27:
        break
cap.release()
cv2.destroyAllWindows()