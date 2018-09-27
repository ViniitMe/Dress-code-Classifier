from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np 
import imutils
import pickle
import cv2
import os

image=cv2.imread("/home/vinit/Downloads/keras-multi-label/examples/example_01.jpg")
image=cv2.imread("/home/vinit/Downloads/blue.jpg")
output=image.copy()

image=cv2.resize(image,(96,96))
image=image.astype("float32")/255.0
image=img_to_array(image)
#making dimension (1,96,96,3)
image=np.expand_dims(image,axis=0)

model=load_model("/home/vinit/Desktop/dress/dress_model.h5py")
lb=pickle.loads(open("/home/vinit/Desktop/dress/label.pickle","rb").read())
#lb.classes_ (it will show the classes)

proba=model.predict(image)[0]
idx=np.argmax(proba)
label=lb.classes_[idx]

output=imutils.resize(output,width=400)
cv2.putText(output,label,(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
print("{}".format(label))
cv2.imshow("output",output)
cv2.waitKey(0) & ord('q')
