import cv2

faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeDetector = cv2.CascadeClassifier("haarcascade_eye.xml")
noseDetector = cv2.CascadeClassifier("haarcascade_nose.xml")
mouthDetector = cv2.CascadeClassifier("haarcascade_mouth.xml")

camera = cv2.VideoCapture(0)

# camera.set(3, 450)
# camera.set(4, 300)

webcamId = 1

while True:
    frameId, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Deteksi wajah
    detection = faceDetector.detectMultiScale(
        gray,
        scaleFactor = 1.5, # Parameter specifying how much the image size is reduced at each image scale.
        minNeighbors = 3, # Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    )
    for x,y,w,h in detection:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,0,255),3)
        cv2.imshow("Detect webcam", frame)

        # Deteksi mata
        video_gray = gray[y:y+h, x:x+w]
        video_color = frame[y:y+h, x:x+w]
        eye = eyeDetector.detectMultiScale(
            video_gray,
            scaleFactor = 1.3,
            minNeighbors = 5
        )
        for ex,ey,ew,eh in eye:
            cv2.rectangle(video_color,(ex,ey),(ex+ew,ey+eh),(255, 0, 0),3)
            cv2.imshow("Detect webcam", frame)

        # Deteksi hidung
        nose = noseDetector.detectMultiScale(
            video_gray,
            scaleFactor = 1.3,
            minNeighbors = 5
        )
        for ex,ey,ew,eh in nose:
            cv2.rectangle(video_color,(ex,ey),(ex+ew,ey+eh),(0, 255, 0),3)
            cv2.imshow("Detect webcam", frame)

        # Deteksi mulut
        mouth = mouthDetector.detectMultiScale(
            gray,
            scaleFactor = 1.3,
            minNeighbors = 5
        )
        for x,y,w,h in mouth:
            cv2.rectangle(frame,(x,y),(x+w, y+h),(0, 255, 255),3)
            cv2.imshow("Detect webcam", frame)

    # cv2.imwrite('webcam/'+'person_'+ str(webcamId) +'.jpg', frame)
    # webcamId+=1

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & webcamId >= 50:
        break

camera.release()
cv2.destroyAllWindows()