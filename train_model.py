'''
Class 0:Honda city,1:Swift Dezire
'''

import numpy as np
import cv2
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.layers import GlobalAveragePooling2D,Flatten,Dense,Input,Dropout
from keras.models import Model,load_model
from keras.optimizers import Adam,RMSprop
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
from sklearn.utils import shuffle
import keras
from albumentations import (
    HorizontalFlip, IAAPerspective, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose,VerticalFlip
)



#Set Path
train_path=r'C:\Users\Anu\Downloads\Python Projects\car_images\train'
car_models=os.listdir(train_path)

#Hyperparameters
BATCH_SIZE=3
DIMS=(224,224,3)
EPOCHS=15

class_dict={car_models[0]:0,car_models[1]:1}

#Load Train Indexes
train_image_indexes=[]
train_targets=[]
for ix,model in enumerate(car_models):
    for file in os.listdir(os.path.join(train_path,model)):
        train_image_indexes.append(os.path.join(train_path,model,file))
        train_targets.append(ix)

train_X,val_X,train_y,val_y=train_test_split(train_image_indexes,train_targets,test_size=0.2)

########################
#DATA GENERATOR
class DataGenerator(Sequence):
    def __init__(self,data_indexes,batch_size=BATCH_SIZE,is_train=True,shuffle=True,dims=DIMS):
        self.image_indexes=data_indexes[0]
        self.targets=data_indexes[1]
        self.batch_size=batch_size
        self.is_train=is_train
        self.shuffle=shuffle
        self.on_epoch_end()
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_indexes))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(len(self.image_indexes)//self.batch_size)
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        img_ids = [self.image_indexes[k] for k in indexes]
        lbls=[self.targets[k] for k in indexes]
        X, y = self.__data_generation(img_ids,lbls)
        return X, y
    
    def augment_flips_color(self,p=.5):
        return Compose([
            RandomRotate90(),
            Transpose(),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
            Blur(blur_limit=3),
            VerticalFlip(),
            HorizontalFlip()
            
        ], p=p)
    
    def __data_generation(self,image_ix,labels):
        tmp=np.zeros((BATCH_SIZE,DIMS[0],DIMS[1],DIMS[2])) 
        aug = self.augment_flips_color(p=1)
        for ix,img_name in enumerate(image_ix):
            img=cv2.imread(img_name)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=img.astype('float')/255.
            img=cv2.resize(img,(DIMS[0],DIMS[1]))
            if self.is_train:
                img = aug(image=img)['image']
            tmp[ix]=img
        return tmp,np.array(labels)
    
train_generator=DataGenerator([train_X,train_y],is_train=True)
val_generator=DataGenerator([val_X,val_y],is_train=False)


##Checkpoint
ckpt_path=r'C:\Users\Anu\Downloads\Python Projects\Output\car_classifier.h5'
mc=ModelCheckpoint(ckpt_path,monitor='val_loss',period=1,save_best_only=True)
    
#Model
base_feat=ResNet50(weights='imagenet',input_shape=DIMS,include_top=False)
x=GlobalAveragePooling2D()(base_feat.output)
x=Dropout(0.4)(x)
out=Dense(1,activation='sigmoid')(x)
model=Model(base_feat.input,out)

model.compile(loss='binary_crossentropy',optimizer=Adam(0.0001),metrics=['acc'])
model_data=model.fit_generator(train_generator,epochs=10,steps_per_epoch=train_generator.__len__(),
                    validation_data=val_generator,validation_steps=val_generator.__len__(),callbacks=[mc])
