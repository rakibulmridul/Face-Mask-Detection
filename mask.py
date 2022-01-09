import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')

print("With mask shape: ",with_mask.shape)
print("Without mask shape: ",without_mask.shape)

#4d to 2d
with_mask=with_mask.reshape(1163,50*50*3)
without_mask=without_mask.reshape(3547,50*50*3)

print("With mask shape: ",with_mask.shape)
print("Without mask shape: ",without_mask.shape)
# print("incorrect mask shape: ",incorrect_mask.shape)
#concating shape 1st  1163 with_mask and last images without mask
x=np.r_[with_mask,without_mask]
print("All image shape: ",x.shape)
labels=np.zeros(x.shape[0])#fill all with 0
labels[1163:]=1.0 #1st 1163=0 and next after 1163=1.0 # its for set the target value(0=mask,1=without mask)
# labels[4710:]=2.0
x_train,x_test,y_train,y_test=train_test_split(x,labels,test_size=0.25)
print("Train shape: ",x_train.shape)
X=x_train

pca=PCA(n_components=20)#dimentin reduction for fast algorithm
x_train=pca.fit_transform(x_train)#Fit the model with x_train and apply the dimensionality reduction on x_train.
print("transformed shape:", x_train.shape)
# pca = PCA().fit(x_train)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');
# plt.show()
X_new = pca.inverse_transform(x_train)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal');
plt.show()
print(x_train[0])
print("After dimention reduction Train shape: ",x_train.shape)
svm=SVC()#Let's build support vector machine model
#history=svm.fit(x_train,y_train)#Train the model using the training sets
svm.fit(x_train,y_train)
x_test=pca.transform(x_test)#Apply dimensionality reduction to x_test
y_pred=svm.predict(x_test)#Predict the response for test dataset
#print(y_pred)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy score: ",accuracy_score(y_test,y_pred))#Accuracy can be computed by comparing actual test set values and predicted values.

cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

faceCascade = cv2.CascadeClassifier(cascPathface)

video_capture = cv2.VideoCapture(0)
data=[]
font=cv2.FONT_HERSHEY_SIMPLEX
while True:
    # Capture frame-by-frame
    flag, img = video_capture.read()
    if flag:
        faces = faceCascade.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h),(0,255,10), 3)
            face=img[y:y+h,x:x+w, : ]
            face=cv2.resize(face,(50,50))
            face=face.reshape(1,-1)
            print(face)
            face=pca.transform(face)
            print(face)
            pred= svm.predict(face)
            if(pred==0):
                cv2.putText(img,"Mask",(x,y),font,1,(244,250,250),2)
                print("mask")
            elif(pred==1):
                cv2.putText(img,"No Mask",(x,y),font,1,(0,0,250),2)
                print("No mask")
            print("Detected")
    cv2.imshow('Face Mask Detector', img)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break
video_capture.release()
cv2.destroyAllWindows()