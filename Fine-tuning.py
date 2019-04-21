
from __future__ import print_function
import keras
from keras.datasets import cifar10,cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Reshape, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam, SGD
#from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
#from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.applications.mobilenet import MobileNet  #MobileNetV2
from keras.applications.densenet import DenseNet121
from keras.applications.nasnet import NASNetMobile

import numpy as np
import os
import shutil
import random
import matplotlib.pyplot as plt
#from keras.utils.visualize_util import 
import cv2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.callbacks import ModelCheckpoint

#from getDataSet import getDataSet

#import h5py

def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))

def treatise(frame,size):
    frame1 = cv2.resize(frame, dsize=size, interpolation=cv2.INTER_CUBIC)
    gridsize=8
    #moto_file=frame #'sora_7_after.png'
    bgr = frame1  #cv2.imread(s1.jpg,1) #k1.jpg s1.jpg
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr   

batch_size = 32
num_classes = 100
epochs = 1
data_augmentation = False #True #False
img_rows=224
img_cols=224
result_dir="./history"

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

size = img_rows, img_cols
X_train =[]
X_test = []
for i in range(50000):
    dst = cv2.resize(x_train[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC) #cv2.INTER_LINEAR #cv2.INTER_CUBIC
    dst = dst[:,:,::-1]
    dst = treatise(dst, size)
    X_train.append(dst)
for i in range(10000):
    dst = cv2.resize(x_test[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    dst = dst[:,:,::-1]  
    dst = treatise(dst, size)
    X_test.append(dst)
X_train = np.array(X_train)
X_test = np.array(X_test)

y_train=y_train[:50000]
y_test=y_test[:10000]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#x_train,y_train,x_test,y_test = getDataSet(img_rows,img_cols)
    #このままだと読み込んでもらえないので、array型にします。
    #x_train = np.array(x_train).astype(np.float32).reshape((len(x_train),3, 32, 32)) / 255
x_train = np.array(X_train)  #/ 255
y_train = np.array(y_train).astype(np.int32)
x_test = np.array(X_test) #/ 255
y_test = np.array(y_test).astype(np.int32)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# VGG16モデルと学習済み重みをロード
# Fully-connected層（FC）はいらないのでinclude_top=False）
input_tensor = Input(shape=x_train.shape[1:])  #(img_rows, img_cols, 3))
#vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
#vgg19 = VGG19(include_top=False, weights='imagenet', input_tensor=input_tensor)
#InceptionV3 = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_tensor)
#ResNet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
#MobileNetV2 = MobileNetV2(input_shape=None, alpha=1.0, depth_multiplier=1, include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
#MobileNet = MobileNet(input_shape=(img_rows, img_cols,3), alpha=1.0, depth_multiplier=1,include_top=False, weights='imagenet', input_tensor=None)
#DenseNet121 = DenseNet121(input_shape=(img_rows, img_cols,3),include_top=False, weights='imagenet', input_tensor=None)
NasNetMobile = NASNetMobile(input_shape=(img_rows, img_cols,3), include_top=False, weights='imagenet', input_tensor=None)

# FC層を構築
top_model = Sequential()
top_model.add(Flatten(input_shape=NasNetMobile.output_shape[1:])) #vgg16,vgg19,InceptionV3,ResNet50,MobileNet,DenseNet121
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))

# VGG16とFCを接続
#model = Model(input=vgg16.input, output=top_model(vgg16.output))
#model = Model(input=vgg19.input, output=top_model(vgg19.output))
#model = Model(input=InceptionV3.input, output=top_model(InceptionV3.output))
#model = Model(input=ResNet50.input, output=top_model(ResNet50.output))
#model = Model(input=MobileNetV2.input, output=top_model(MobileNetV2.output))
#model = Model(input=MobileNet.input, output=top_model(MobileNet.output))
#model = Model(input=DenseNet121.input, output=top_model(DenseNet121.output))
model = Model(input=NasNetMobile.input, output=top_model(NasNetMobile.output))

# 最後のconv層の直前までの層をfreeze
#trainingするlayerを指定　VGG16では18,15,10,1など 20で全層固定
#trainingするlayerを指定　VGG16では16,11,7,1など 21で全層固定
#trainingするlayerを指定　InceptionV3では310で全層固定
#trainingするlayerを指定　ResNet50では174で全層固定
#trainingするlayerを指定　MobileNetV2では  で全層固定
for layer in model.layers[1:1]:  
    layer.trainable = False

# Fine-tuningのときはSGDの方がよい⇒adamがよかった
lr = 0.00001
opt = keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=1e-6)
#opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# モデルのサマリを表示
model.summary()
#plot(model, show_shapes=True, to_file=os.path.join(result_dir,
#model.load_weights('params_model_VGG16L3_i_190.hdf5')

for i in range(epochs):
    epoch=100
    if not data_augmentation:
        print('Not using data augmentation.')

        # 学習履歴をプロット
        #plot_history(history, result_dir)
        checkpointer = ModelCheckpoint(filepath='./cifar10/weights_only_cifar100_NasNetMobile_224.hdf5', 
                                       monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True)
        early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max',
                                       verbose=1)
        lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5,
                                         factor=0.5, min_lr=0.00001, verbose=1)
        csv_logger = CSVLogger('history_cifar100_NasNetMobile_224.log', separator=',', append=True)
        callbacks = [early_stopping, lr_reduction, csv_logger,checkpointer]

        history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epoch,
                  callbacks=callbacks,          
                  validation_data=(x_test, y_test),
                  shuffle=True)        
        
        # save weights every epoch
        model.save_weights('params_NasNetMobile_epoch_{0:03d}.hdf5'.format(i), True)   
        save_history(history, os.path.join(result_dir, 'history_NasNetMobile_epoch_{0:03d}.txt'.format(i)))
        
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        checkpointer = ModelCheckpoint(filepath='./cifar10/weights_only_cifar100_NasNetMobile_224.hdf5', 
                                       monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True)
        early_stopping = EarlyStopping(monitor='val_acc', patience=10, mode='max',
                                       verbose=1)
        lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5,
                                         factor=0.5, min_lr=0.000001, verbose=1)
        csv_logger = CSVLogger('history_cifar100_NasNetMobile_224.log', separator=',', append=True) 
        callbacks = [early_stopping, lr_reduction, csv_logger,checkpointer]

        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epoch,
                            callbacks=callbacks,          
                            validation_data=(x_test, y_test))        
        model.save_weights('params_NasNetMobile_epoch_{0:03d}.hdf5'.format(i), True)   
        save_history(history, os.path.join(result_dir, 'history_NasNetMobile_epoch_{0:03d}.txt'.format(i)))
