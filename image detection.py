import cv2

# Load pre-trained face detection data
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load image
img = cv2.imread(r'C:\Users\Pavan\OneDrive\Desktop\facerecognitiondetection\models\Prabhas.png')

# Check if the image was loaded successfully
if img is None:
    print("Error: Image not found or could not be loaded.")
else:
    # Convert to grayscale
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    if len(face_coordinates) > 0:
        # Draw rectangles around detected faces
        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 200, 0), 10)

        # Display the image with detected faces
        cv2.imshow('Face Detection', img)
        cv2.waitKey(0)  # Wait indefinitely for a key press
    else:
        print("No faces detected.")

cv2.destroyAllWindows()
