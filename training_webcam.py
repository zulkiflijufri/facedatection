import cv2, os, numpy as np
from PIL import Image

faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImage(directory):
    imagesDir = [os.path.join(directory,f) for f in os.listdir(directory)] # ambil gambar sesuai directory
    webcamImg = []
    webcamId = []
    for imgDir in imagesDir:
        grayImg = Image.open(imgDir).convert('L') # 'L' => gray scale
        grayImg = np.array(grayImg)
        grayId = int(os.path.split(imgDir)[-1].split("_")[1].split(".")[0])
        webcam = faceDetector.detectMultiScale(grayImg)
        for x,y,w,h in webcam:
            webcamImg.append(grayImg[y:y+h, x:x+w])
            webcamId.append(grayId)
    return webcamImg, webcamId

faceRecog = cv2.face.LBPHFaceRecognizer_create()
faces, faceId = getImage('webcam')
faceRecog.train(faces, np.array(faceId))
faceRecog.write('training/webcam.xml')