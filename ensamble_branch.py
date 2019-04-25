from __future__ import print_function
import keras
from keras.datasets import cifar10,cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers import Input, Reshape, Embedding,Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
#from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
#from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
#from keras.applications.mobilenet import MobileNet  #MobileNetV2
#from keras.applications.densenet import DenseNet121
#from keras.applications.nasnet import NASNetMobile

import numpy as np
import os
import shutil
import random
import matplotlib.pyplot as plt
#from keras.utils.visualize_util import 
import cv2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.callbacks import ModelCheckpoint

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

from keras.models import Model
from keras.layers import Input, Dense, Add, Multiply

def build(input, *nodes):
    x = input
    for node in nodes:
        if callable(node):
            x = node(x)
        elif isinstance(node, list):
            x = [build(x, branch) for branch in node]
        elif isinstance(node, tuple):
            x = build(x, *node)
        else:
            x = node
    return x

def example_1():
    input = Input((10,))
    output = build(
        input,
        Dense(10),
        [Dense(11, activation='relu'), Dense(11, activation='relu')],
        Add(),
        [Dense(12, activation='relu'), Dense(12, activation='relu')],
        Multiply(),
        )
    model = Model(input, output)
    model.summary()
    
from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, Concatenate
def example_multi_input_and_multi_output():
    main_input = Input(shape=(100,), dtype='int32', name='main_input')
    auxiliary_input = Input(shape=(5,), name='aux_input')
    outputs = build(
        main_input,
        Embedding(output_dim=512, input_dim=10000, input_length=100),
        LSTM(32),
        [Dense(1, activation='sigmoid', name='aux_output'),
         ([auxiliary_input, lambda x: x],
          Concatenate(),
          Dense(64, activation='relu'),
          Dense(64, activation='relu'),
          Dense(64, activation='relu'),
          Dense(1, activation='sigmoid', name='main_output')
         )
        ]
    )
    model = Model([main_input, auxiliary_input], outputs)
    model.summary()

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
def Wide_vgg16(img_rows, img_cols,num_classes):
    input = Input(shape=(img_rows, img_cols, 3))
    output = build(
        input,
        [Conv2D(64, (3, 3), padding='same'),
         Conv2D(64, (3, 3), padding='same')],
        Add(),
        Activation('relu'),
        [Conv2D(64, (3, 3), padding='same'),
         Conv2D(64, (3, 3), padding='same')],
        Add(),
        Activation('relu'),
        BatchNormalization(axis=3),
        Dropout(0.5),
        AveragePooling2D((2, 2)),
        [Conv2D(128, (3, 3), padding='same'),
         Conv2D(128, (3, 3), padding='same')],
        Add(),
        Activation('relu'),
        [Conv2D(128, (3, 3), padding='same'),
         Conv2D(128, (3, 3), padding='same')],
        Add(),
        Activation('relu'),
        BatchNormalization(axis=3),
        Dropout(0.5),
        AveragePooling2D((2, 2)),
        [Conv2D(256, (3, 3), padding='same'),
         Conv2D(256, (3, 3), padding='same'),
         Conv2D(256, (3, 3), padding='same')],
        Add(),
        Activation('relu'),
        [Conv2D(256, (3, 3), padding='same'),
         Conv2D(256, (3, 3), padding='same'),
         Conv2D(256, (3, 3), padding='same')],
        Add(),
        Activation('relu'),
        [Conv2D(256, (3, 3), padding='same'),
         Conv2D(256, (3, 3), padding='same'),
         Conv2D(256, (3, 3), padding='same')],
        Add(),
        Activation('relu'),
        BatchNormalization(axis=3),
        Dropout(0.5),
        AveragePooling2D((2, 2)),
        [Conv2D(512, (3, 3), padding='same'),
         Conv2D(512, (3, 3), padding='same'),
         Conv2D(512, (3, 3), padding='same')],
        Add(),
        Activation('relu'),
        [Conv2D(512, (3, 3), padding='same'),
         Conv2D(512, (3, 3), padding='same'),
         Conv2D(512, (3, 3), padding='same')],
        Add(),
        Activation('relu'),
        [Conv2D(512, (3, 3), padding='same'),
         Conv2D(512, (3, 3), padding='same'),
         Conv2D(512, (3, 3), padding='same')],
        Add(),
        Activation('relu'),
        Dropout(0.5),
        Flatten(),
        Dense(num_classes, activation='softmax')
    )

    model = Model(input, output)
    model.summary()
    return model

def Double_vgg16(img_rows, img_cols,num_classes):
    input = Input(shape=(img_rows, img_cols, 3))
    output1 = build(
        input,
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(axis=3),
        Dropout(0.75),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(axis=3),
        Dropout(0.75),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(axis=3),
        Dropout(0.75),
        MaxPooling2D((2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(axis=3),
        Dropout(0.75),
        MaxPooling2D((2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(axis=3),
        Dropout(0.75),
        MaxPooling2D((2, 2))
    )
    output2 = build(
        input,
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(axis=3),
        Dropout(0.75),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(axis=3),
        Dropout(0.75),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(axis=3),
        Dropout(0.75),
        MaxPooling2D((2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(axis=3),
        Dropout(0.75),
        MaxPooling2D((2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(axis=3),
        Dropout(0.75),
        MaxPooling2D((2, 2))
    )
    
    output3 = build(
        [output1, output2],
        Add(),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.75),
        Dense(num_classes, activation='softmax')
    )

    model = Model(input, output3)
    model.summary()
    return model

def example_vgg16_1(img_rows, img_cols,num_classes):
    input = Input(shape=(img_rows, img_cols, 3))
    output = build(
        input,
        Conv2D(64, (3, 3), padding='same'),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(axis=3),
        Dropout(0.75),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), padding='same'),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(axis=3),
        Dropout(0.75),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), padding='same'),
        Conv2D(256, (3, 3), padding='same'),
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(axis=3),
        Dropout(0.75),
        MaxPooling2D((2, 2))
    )
    x=Flatten()(output)
    x=Dense(256, activation='relu')(x)
    x=Dropout(0.75)(x)
    x=Dense(num_classes, activation='softmax')(x)
    model0=Model(input, x)
    model0.summary()
    model = Model(input, output)
    model.summary()
    return model0, model

def example_vgg16_2(img_rows, img_cols,num_classes):
    input = Input(shape=(img_rows, img_cols, 256))
    output = build(
        input,
        Conv2D(512, (3, 3), padding='same'),
        Conv2D(512, (3, 3), padding='same'),
        Conv2D(512, (3, 3), padding='same'),
        BatchNormalization(axis=3),
        Dropout(0.75),
        MaxPooling2D((2, 2)),
        Conv2D(512, (3, 3), padding='same'),
        Conv2D(512, (3, 3), padding='same'),
        Conv2D(512, (3, 3), padding='same'),
        BatchNormalization(axis=3),
        Dropout(0.75),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.75),
        Dense(num_classes, activation='softmax')
    )

    model = Model(input, output)
    model.summary()
    return model
    
batch_size = 16
num_classes = 10
epochs = 10
data_augmentation = False #True #False
img_rows, img_cols=64,64

result_dir="./history"

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

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

# Fine-tuningのときはSGDの方がよい⇒adamがよかった
lr = 0.00001
opt = keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=1e-6)
#opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9)
"""
model0,model1 = example_vgg16_1(img_rows, img_cols,num_classes)
model2 = example_vgg16_2(8,8,num_classes)
model = Model(input=model1.input, output=model2(model1.output))
model0 = Model(input=model1.input, output=model0.output)
"""
model = Double_vgg16(img_rows, img_cols,num_classes) #Wide_vgg16  #example_vgg16 example_residual_connection
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
"""
model0.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
"""
print('Not using data augmentation.')

# 学習履歴をプロット
#plot_history(history, result_dir)
checkpointer = ModelCheckpoint(filepath='./cifar100/weights_only_cifar10_example_vgg16_64-ol.hdf5', 
                          monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_acc', patience=50, mode='max',
                          verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=50,
                          factor=0.5, min_lr=0.000001, verbose=1)
csv_logger = CSVLogger('./cifar100/history_cifar10_example_vgg16_64-ol.log', separator=',', append=True)
callbacks = [early_stopping, lr_reduction, csv_logger,checkpointer]

history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=100,
                  callbacks=callbacks,          
                  validation_data=(x_test, y_test),
                  shuffle=True)        
# save weights every epoch

model.save_weights('./cifar100/params_example_vgg16_ol_epoch_{0:03d}.hdf5'.format(0), True)   
save_history(history, os.path.join(result_dir, 'history_example_vgg16_ol_epoch_{0:03d}.txt'.format(0)))
"""
checkpointer = ModelCheckpoint(filepath='./cifar100/weights_only_cifar10_example_vgg16_64-3.hdf5', 
                          monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_acc', patience=50, mode='max',
                          verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=50,
                          factor=0.5, min_lr=0.000001, verbose=1)
csv_logger = CSVLogger('./cifar100/history_cifar10_example_vgg16_64-3.log', separator=',', append=True)
callbacks = [early_stopping, lr_reduction, csv_logger,checkpointer]

history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=100,
                  callbacks=callbacks,          
                  validation_data=(x_test, y_test),
                  shuffle=True)        
# save weights every epoch
model.save_weights('./cifar100/params_example_vgg16_epoch_{0:03d}.hdf5'.format(0), True)   
save_history(history, os.path.join(result_dir, 'history_example_vgg16_epoch_{0:03d}.txt'.format(0)))
"""