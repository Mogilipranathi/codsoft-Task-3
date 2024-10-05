import cv2
 #Load the face, eye, smile, and mouth classifiers

fullbody_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Define the face detector function
def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        x = x - 50
        w = w + 50
        y = y - 50
        h = h + 50
        x = max(x, 0)
        y = max(y, 0)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
        face_roi = img[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_roi, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    return img

# Start video capture from the default camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    processed_frame = face_detector(frame)
    cv2.imshow('FRAME', processed_frame)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
