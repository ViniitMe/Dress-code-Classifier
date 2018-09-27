from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np 
import imutils
import pickle
import cv2
import os

model=load_model("model.h5py")
lb=pickle.loads(open("lb.pickle","rb").read())
#lb=['black_jeans', 'blue_dress', 'blue_jeans', 'blue_shirt', 'red_dress', 'red_shirt']

cap=cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FPS,1e-1000)
while(1):
	ret,frame=cap.read()
	output=frame.copy()

	frame=cv2.resize(frame,(96,96))
	frame=frame.astype("float32")/255.0
	frame=img_to_array(frame)
	frame=np.expand_dims(frame,axis=0)

	#lb=pickle.loads(open("/home/vinit/Desktop/dress/label.pickle","rb").read())
	proba=model.predict(frame)[0]
	idx=np.argmax(proba)
	label=lb.classes_[idx]
	#label=lb[idx]
	label="{}: {:.2f}%".format(label,proba[idx]*100)
	output=imutils.resize(output,width=400)
	
	if proba[idx] > 0.9:
	        cv2.putText(output,label,(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
	#print("{}: {:.2f}%".format(label,proba[idx]*100))
	cv2.imshow("Output",output)

	if cv2.waitKey(1) & 0xFF==ord('q'):
		break

cv2.destroyAllWindows()
cap.release()
