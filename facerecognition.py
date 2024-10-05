import cv2

# Open the video capture; use 0 for the default camera or the correct index for other cameras
video_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, img = video_capture.read()
    
    # Check if the frame was captured successfully
    if not ret:
        print("Failed to capture image")
        break

    # Display the image in a window
    cv2.imshow("Face Detection", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
