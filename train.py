import numpy as np 
import keras
from keras.models import Sequential,Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from sklearn import cross_validation
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.layers.advanced_activations import LeakyReLU
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
import imutils
import pickle
import cv2
import os
import random
from keras.utils import to_categorical

image_dim=(96,96,3)

path="/media/vinit/120eed49-b701-489b-a66b-7e4315851eb6/vinit/dress/dataset"
imagepaths=sorted(list(paths.list_images(path)))
random.shuffle(imagepaths)

data=[]
labels=[]
for i in imagepaths:
	image=cv2.imread(i)
	image=cv2.resize(image,(image_dim[1],image_dim[0]))
	image=img_to_array(image)
	data.append(image)
	label=i.split(os.path.sep)[-2]
	labels.append(label)

data=np.array(data,dtype='float32')/255.0
labels=np.array(labels)

lb=LabelBinarizer()
labels=lb.fit_transform(labels)

x_train,x_test,y_train,y_test=cross_validation.train_test_split(data,labels,test_size=0.2,random_state=42)
aug=ImageDataGenerator(rotation_range=25,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,horizontal_flip=True,fill_mode="nearest")
num_classes=len(lb.classes_)
model=Sequential()

#CONV => RELU => POOL
model.add(Conv2D(32,(3,3),padding="same",activation="linear",input_shape=(96,96,3)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization(axis=-1)) #axis=channel_dim
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))

#(CONV => RELU) *2 => POOL
model.add(Conv2D(64,(3,3),padding="same",activation="linear"))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization(axis=-1)) #axis=channel_dim
model.add(Conv2D(64,(3,3),padding="same",activation="linear"))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#(CONV => RELU) *2 => POOL
model.add(Conv2D(128,(3,3),padding="same",activation="linear"))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization(axis=-1)) #axis=channel_dim
model.add(Conv2D(128,(3,3),padding="same",activation="linear"))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024,activation="linear"))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes,activation="softmax"))

print("Compiling model....")
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()

train=model.fit_generator(aug.flow(x_train,y_train,batch_size=64),validation_data=(x_test,y_test),steps_per_epoch=len(x_train)/64,epochs=50,verbose=1)

model.save("model.h5py")
f=open("lb.pickle","wb")
f.write(pickle.dumps(lb))
f.close()

#Saving a model which predicts best(increased accuracy model)
#make validation set:
#x_valid=x_train[200:]  #20% approx
#y_valid=y_train[200:]
#x_train=x_train[:200]
#y_train=y_train[:200] 
#from keras.callbacks import ModelCheckPoint
#model.compile(loss='categorical_crossentropy',optimizer='nadam',metrics=['accuracy'])
#checkpoint=ModelCheckpoint(filepath='model_best.hdf5',save_best_only=True,verbose=1)
#train = model.fit(X_train, y_train, batch_size=16, epochs=40, validation_data=(X_valid, y_valid), verbose=2, callbacks=[checkpoint])

