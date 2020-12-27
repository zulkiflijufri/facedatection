import cv2

face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye = cv2.CascadeClassifier("haarcascade_eye.xml")
nose = cv2.CascadeClassifier("haarcascade_nose.xml")
mouth = cv2.CascadeClassifier("haarcascade_mouth.xml")

img = cv2.imread('obama.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detection = face.detectMultiScale(
    gray,
    scaleFactor = 1.05,
    minNeighbors = 5
)

# Deteksi wajah
for x, y, w, h in detection:
    cv2.rectangle(
        img,
        (x,y),
        (x+w, y+h),
        (0, 0, 255),
        3,
    )
    cv2.imshow('Detect Image', img)
    cv2.waitKey(0)
    print("Wajah")
    print(x,y,w,h)

    # Deteksi mata
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye.detectMultiScale(
        roi_gray
    )
    for ex,ey,ew,eh in eyes:
        cv2.rectangle(
            roi_color,
            (ex,ey),
            (ex+ew,ey+eh),
            (255, 0, 0),
            3
        )
        cv2.imshow('Detect Image', img)
        cv2.waitKey(0)
        print("Mata")
        print(ex,ey,ew,eh)
    # Deteksi hidung
    noses = nose.detectMultiScale(
        roi_gray,
        scaleFactor = 1.6,
        minNeighbors = 5
    )
    for ex,ey,ew,eh in noses:
        cv2.rectangle(
            roi_color,
            (ex,ey),
            (ex+ew,ey+eh),
            (0, 255, 0),
            3
        )
        cv2.imshow('Detect Image', img)
        cv2.waitKey(0)
        print("Hidung")
        print(ex,ey,ew,eh)
    # Deteksi mulut
    mouths = mouth.detectMultiScale(
        gray,
        scaleFactor = 1.6,
        minNeighbors = 5
    )
    for x,y,w,h in mouths:
        cv2.rectangle(
            img,
            (x,y),
            (x+w, y+h),
            (0, 0, 0),
            3,
        )
        cv2.imshow('Detect Image', img)
        cv2.waitKey(0)
        print("Hidung")
        print(x,y,w,h)

resized = cv2.resize(img, (600,600))
cv2.imwrite('hasil_deteksi.jpg', img)
cv2.destroyAllWindows()