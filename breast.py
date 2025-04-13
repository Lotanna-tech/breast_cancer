import PIL
#import tensorflow as tf
from tensorflow import keras
import pathlib
import PIL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
import tf_keras
data_dir=pathlib.Path('Dataset')
data=list(data_dir.glob('benign/*'))

#image=PIL.Image.open(str(data[0]))
#plt.imshow(image)
#plt.show()

data_dict={
    'benign':list(data_dir.glob('benign/*')),
    'malignant':list(data_dir.glob('malignant/*')),
    'normal':list(data_dir.glob('normal/*'))
}

data_label={
    'benign':0,
    'malignant':1,
    'normal':2,

   
}
X=[]
y=[]
for breast_data,images in data_dict.items():
    for image in images:
        img=cv2.imread(image)
        img=cv2.resize(img,(224,224))
        X.append(img)
        y.append(data_label[breast_data])

link="https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feauture_extractor=hub.KerasLayer(link,trainable=False,input_shape=(224,224,3))

X=np.array(X)
y=np.array(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

X_train=X_train/255
X_test=X_test/255

model=tf_keras.Sequential([
    tf_keras.layers.InputLayer(input_shape=(224, 224, 3)),

    feauture_extractor,
    tf_keras.layers.Dense(3,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.fit(X_train,y_train,epochs=1)