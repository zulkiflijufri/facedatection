import cv2

faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeDetector = cv2.CascadeClassifier("haarcascade_eye.xml")
noseDetector = cv2.CascadeClassifier("haarcascade_nose.xml")
mouthDetector = cv2.CascadeClassifier("haarcascade_mouth.xml")

faceRecog = cv2.face.LBPHFaceRecognizer_create()
faceRecog.read('training/webcam.xml')

camera = cv2.VideoCapture(0)

camera.set(3, 640)
camera.set(4, 480)

peoples = ["Siapa ya?", "Zulkifli"]

minWidht = camera.get(3)
minHeight = camera.get(4)

while True:
    _, frame = camera.read()
    # frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Deteksi wajah
    face = faceDetector.detectMultiScale(
        gray,
        scaleFactor = 1.5, # Parameter specifying how much the image size is reduced at each image scale.
        minNeighbors = 3, # Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        # minSize = (640,480)
    )
    for x,y,w,h in face:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,0,255),3)
        _, confidence  = faceRecog.predict(gray[y:y+h, x:x+w])
        # print(100 - round(confidence,2))
        if (confidence <= 50):
            name = peoples[1]
            numConfidence = "{conf}%".format(conf = round(confidence,2))
        else:
            name = peoples[0]
            numConfidence = "{conf}%".format(conf = round(confidence,2))
        cv2.putText(frame, name, (x+w-h+25,y+h-w-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, numConfidence, (x+w-h+25,y+h-w-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Face Detection", frame)

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
            cv2.imshow("Face Detection", frame)

        # Deteksi hidung
        nose = noseDetector.detectMultiScale(
            video_gray,
            scaleFactor = 1.3,
            minNeighbors = 5
        )
        for ex,ey,ew,eh in nose:
            cv2.rectangle(video_color,(ex,ey),(ex+ew,ey+eh),(0, 255, 0),3)
            cv2.imshow("Face Detection", frame)

        # Deteksi mulut
        mouth = mouthDetector.detectMultiScale(
            gray,
            scaleFactor = 1.3,
            minNeighbors = 5
        )
        for x,y,w,h in mouth:
            cv2.rectangle(frame,(x,y),(x+w, y+h),(0, 255, 255),3)
            cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()