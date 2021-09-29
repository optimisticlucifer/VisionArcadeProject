# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:32:46 2021

@author: Murtaza Hasan
"""

  
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Flatten,Conv2D,MaxPooling2D,Dense,Dropout
from keras.optimizers import Adam
# Step 1 - Building the CNN

model=Sequential()
model.add(Conv2D(30,(5,5),activation="relu",input_shape=(64,64, 1)))
model.add(Conv2D(30,(5,5),activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(30,(5,5),activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.7))
model.add(Flatten())
model.add(Dense(100,activation="relu"))
model.add(Dense(9,activation="softmax"))# categorical_crossentropy for more than 2



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,width_shift_range=0.1,height_shift_range=0.15,rotation_range=10)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(64, 64),
                                                 batch_size=7,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(64, 64),
                                            batch_size=7,
                                            color_mode='grayscale',
                                            class_mode='categorical') 

model.compile(Adam(lr=0.001),loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(
        training_set,# No of images in training set
        batch_size=7,
        epochs=11,
        validation_data=test_set,)# No of images in test set


# Saving the model
model_json = model.to_json()
with open("model-bw1.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('model-bw1.h5')