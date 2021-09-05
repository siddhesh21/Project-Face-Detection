import cv2
import time
from random import randrange

# Loading some pre-trained data on face frontals from
# opencv (by using haar cascade algorithm)

trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
#img = cv2.imread('Image/avengers-endgame-world-premiere.jpg')


# To Capture video from webcam
webcam = cv2.VideoCapture(0)

# Iterate until video ends or interupt
while True:

    # Reading the current frame
    successful_frame_read, frame = webcam.read()

    # Converting image to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y),
                      (x+w, y+h),
                      (randrange(256),
                       randrange(256),
                       randrange(256)), 5)

    # Display the image where face is spotted
    cv2.imshow("Detect the face", frame)

    # pause compiled program until key press
    key_press = cv2.waitKey(1)

    # Stop App if ESC key is pressed
    if key_press == 27:
        break

# Stop the VideoCapture object
webcam.release()

print("Code Rendered")
print("Code Completed")
