import numpy as np
import cv2
import os
import glob
cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
#FaceMaskDetector\image\
path =glob.glob(r"F:\archive\data\without_mask/*.jpg")
faceCascade = cv2.CascadeClassifier(cascPathface)
video_capture = cv2.VideoCapture(0)
ImageData = []
font=cv2.FONT_HERSHEY_SIMPLEX
count=0
for file in path:
    # Capture frame-by-frame
    image = cv2.imread(file)
    faces = faceCascade.detectMultiScale(image)
    for x,y,w,h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h),(0,255,10), 2)
        face=image[y:y+h,x:x+w, : ]
        face=cv2.resize(face,(50,50))
        ImageData.append(face)
        print(count)

    count=count+1
    cv2.imshow('Face Mask Detector', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#video_capture.release()
#video_capture.release()
cv2.destroyAllWindows()
np.save('without_mask',ImageData)