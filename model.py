import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Dropout, Flatten
from keras.optimizers import Adam
########################################

path='data_set_mask_full'
images=[]
classNo=[]
testRatio=0.2
valRatio=0.2
imgDimension=(32,32,3)

#########################################

myList=os.listdir(path) #returns the list of all files and directories in the specified path.

numOfClasses=len(myList)


print("Importing Classes..........")
for x in range(0, numOfClasses):
	myPicList=os.listdir(path+"/"+str(x))
	# read and resize images (32x32)
	for y in myPicList:
		curImg=cv2.imread(path+"/"+str(x)+"/"+y)
		curImg=cv2.resize(curImg,(imgDimension[0],imgDimension[1]))
		images.append(curImg)
		# 0 => unmasked 1=> masked 
		classNo.append(x)
	print(x)
# convert to numpy array
images=np.array(images)
classNo=np.array(classNo)




#########Splitting The Data into training data and test data###########

x_train, x_test, y_train, y_test=train_test_split(images, classNo, test_size=testRatio) #20%

#########Splitting The Data into training data and validation data###########

x_train, x_validation, y_train, y_validation=train_test_split(x_train, y_train, test_size=valRatio) # 20%




def preprocessing(img):
	img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img=cv2.equalizeHist(img) #  => improves the contrast  & make the image clearer
	img=img/255 # divided by the highest intensity value
	return img


x_train=np.array(list(map(preprocessing, x_train)))
x_test=np.array(list(map(preprocessing, x_test)))
x_validation=np.array(list(map(preprocessing, x_validation)))





x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1)
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)
x_validation=x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2],1)

"""
ImageDataGenerator => fournir une augmentation des données en temps réel. 
Càd il génère des images augmentées à la volée alors que votre modèle est encore en phase d'apprentissage
"""

dataGen=ImageDataGenerator(
	width_shift_range=0.1,
	height_shift_range=0.1,
	zoom_range=0.2,
	shear_range=0.1,
	rotation_range=10)

dataGen.fit(x_train)
#to_categorical =>
y_train=to_categorical(y_train, numOfClasses)
y_test=to_categorical(y_test, numOfClasses)
y_validation=to_categorical(y_validation, numOfClasses)


def myModel():
	sizeOfFilter1=(3,3)
	sizeOfFilter2=(3,3)
	sizeOfPool=(2,2)

	model=Sequential()
	model.add((Conv2D(32, sizeOfFilter1, input_shape=(imgDimension[0],imgDimension[1],1),activation='relu'))) #relu allows only positive values
	model.add((Conv2D(32, sizeOfFilter1,activation='relu')))
	model.add(MaxPooling2D(pool_size=sizeOfPool))

	model.add((Conv2D(64, sizeOfFilter2,activation='relu')))
	model.add((Conv2D(64, sizeOfFilter2,activation='relu')))
	model.add(MaxPooling2D(pool_size=sizeOfPool))
	model.add(Dropout(0.5))


	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.3)) # drop 30% of neurons in each epoch
	model.add(Dense(numOfClasses, activation='softmax')) # last layer
	model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
	return model

model=myModel()
print(model.summary())

history=model.fit(x_train, y_train,batch_size=50,
	epochs=100,
	validation_data=(x_validation,y_validation),
	shuffle=1)
model.save("MyModel2.h5")
print(model.evaluate(x_test,y_test))
