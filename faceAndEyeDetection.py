import cv2
import numpy as np

cap = cv2.VideoCapture(0)

#Load in the Haar Cascades
#The argument passed is the path to the Cascade Classifier
#These classifiers are pre trained, meaning they know what to look for when detecting face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()
    
    #Convert the video image to grayscale so the algorithm can work
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #This will return the position of the face/faces found by the Haar cascade
    #See this link to understand the parameters : 
    #https://stackoverflow.com/questions/20801015/recommended-values-for-opencv-detectmultiscale-parameters
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Draw a rectangle around the face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        #Now get the area of the face to determine where the eyes are
        #ROI = Region of Interest
        #Getting the face from the grayscale image
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #Now look in the grayscale image for the eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        #Draw the eyes
        #We draw the eyes on roi_color because it is relative to the face
        #If we were to draw it on the original image, the eyes would be in the wrong spot
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()