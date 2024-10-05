import cv2


# load pre trained data

trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img = cv2.imreadimg = cv2.imread(r'C:\Users\Pavan\OneDrive\Desktop\facerecognition&detection\prabhas.png')


grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)
(x,y,w,h) = face_coordinates[0]
cv2.rectangle(img,(x,y),(x+w,y+h) , (0,255,0),10)
cv2.imshow('unqcoder',img)
cv2.waitKey(1)

 
 