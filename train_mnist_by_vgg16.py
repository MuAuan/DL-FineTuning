# coding:utf-8
#以下の技術を利用させていただきました
#https://qiita.com/koshian2/items/04853466d77bab360c9d

import numpy as  np
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow.keras.backend as K
from keras import backend as K
import keras
from keras.datasets import mnist
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.callbacks import ModelCheckpoint

from keras.layers import *
from keras.models import Model, Sequential
import cv2
from keras.datasets import cifar10,cifar100
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.datasets import mnist

#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#x_train,y_train,x_test,y_test = getDataSet(img_rows,img_cols)

img_rows, img_cols, ch=128,128,1
num_classes = 10
# データをロード
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 前処理

X_train =[]
X_test = []
for i in range(50000):
    dst = cv2.resize(x_train[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    #dst = dst[:,:,::-1]  
    X_train.append(dst)
for i in range(10000):
    dst = cv2.resize(x_test[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    #dst = dst[:,:,::-1]  
    X_test.append(dst)

X_train = np.array(X_train).reshape(50000,img_rows, img_cols,ch)
X_test = np.array(X_test).reshape(10000,img_rows, img_cols,ch)

y_train=y_train[:50000]
y_test=y_test[:10000]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

x_train = X_train.astype('float32')
x_test = X_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def create_normal_model():
    #input_tensor=(img_rows, img_cols,ch)
    input_tensor = x_train.shape[1:] 
    input_model = Sequential()
    input_model.add(InputLayer(input_shape=input_tensor))
    #input_model.add(GaussianNoise(0.01))
    input_model.add(Conv2D(3, (3, 3),activation='relu', padding='same'))
    # Fully-connected層（FC）はいらないのでinclude_top=False）
    model = VGG16(include_top=False,weights=None, input_tensor=input_model.output)    #weights='imagenet'
    #model = VGG16(include_top=False, input_shape=input_tensor, weights="imagenet")
    #x = GlobalAveragePooling2D()(model.layers[-1].output)
    x = Flatten()(model.layers[-1].output)
    x = Dense(num_classes, activation="softmax")(x)
    return Model(model.inputs, x)

def create_batch_norm_model():
    model = create_normal_model()
    for i, layer in enumerate(model.layers):
        if i==0:
            input = layer.input
            x = input
        else:
            if "conv" in layer.name:
                layer.activation = activations.linear
                x = layer(x)
                x = BatchNormalization()(x)
                x = Activation("relu")(x)
                #x = Dropout(0.5)(x)
            else:
                x = layer(x)

    bn_model = Model(input, x)
    return bn_model

model=create_batch_norm_model()
model.summary()

# Fine-tuningのときはSGDの方がよい⇒adamがよかった
lr = 0.00001
opt = keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=1e-6)
#opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

class Check_layer(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        model.save_weights('./cifar10/mnist_cnn128_'+str(epoch)+'_g.hdf5', True) 
        check_layer(img=x_test[1],epoch=epoch)

def check_layer(img=x_test[1],epoch=0):
    predictions = model.predict(img.reshape(1,128,128,1))
    index_predict=np.argmax(predictions)
    print('Predicted class:',index_predict)
    print(' Probability: {}'.format(predictions[0][index_predict]))

model.load_weights('vgg16_weights.hdf5',by_name=True)
    
ch_layer = Check_layer()
callbacks = [ch_layer] 
history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=20,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=(x_test, y_test))

checkpointer = ModelCheckpoint(filepath='./cifar10/mnist_cnn_128.hdf5', 
                               monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_acc', patience=20, mode='max',
                               verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5,
                               factor=0.5, min_lr=0.000001, verbose=1)
csv_logger = CSVLogger('./cifar10/history_mnist_cnn_128.log', separator=',', append=True)
callbacks = [early_stopping, lr_reduction, csv_logger,checkpointer]

#Learning ; Original x_train, y_train
history = model.fit(x_train, y_train,
          batch_size=64,
          epochs=100,
          callbacks=callbacks,          
          validation_data=(x_test, y_test),
          shuffle=True) 
          
          
