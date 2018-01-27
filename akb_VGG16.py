'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 32, 32, 3)         0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 32, 32, 64)        1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 32, 32, 64)        36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 16, 16, 64)        0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 16, 16, 128)       73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 16, 16, 128)       147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 8, 8, 128)         0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 8, 8, 256)         295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 8, 8, 256)         590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 8, 8, 256)         590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 4, 4, 256)         0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 4, 4, 512)         1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 4, 4, 512)         2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 4, 4, 512)         2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 2, 2, 512)         0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 2, 2, 512)         2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 2, 2, 512)         2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 2, 2, 512)         2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 1, 1, 512)         0
_________________________________________________________________
sequential_1 (Sequential)    (None, 10)                133898
=================================================================
Total params: 14,848,586
Trainable params: 7,473,482
Non-trainable params: 7,375,104
_________________________________________________________________
for layer in model.layers[7:15]:    layer.trainable = False
1562/1562 [==============================] - 76s - loss: 4.5619e-04 - acc: 0.9999 - val_loss: 1.5312 - val_acc: 0.8400
for layer in model.layers[1:10]:    layer.trainable = False
1562/1562 [==============================] - 78s - loss: 0.0024 - acc: 0.9994 - val_loss: 1.0931 - val_acc: 0.8953
i, ir=  110 8.589934592000007e-06
for layer in model.layers[1:1]:    layer.trainable = False
1562/1562 [==============================] - 104s - loss: 0.0096 - acc: 0.9974 - val_loss: 0.7429 - val_acc: 0.9002
=================================================================
for layer in model.layers[1:18]
Total params: 14,848,586
Trainable params: 133,898
Non-trainable params: 14,714,688
_________________________________________________________________
i, ir=  10 8e-05
Using real-time data augmentation.
Epoch 1/1
1562/1562 [==============================] - 29s 19ms/step - loss: 1.2675 - acc: 0.5576 - val_loss: 1.2088 - val_acc: 0.5753

for layer in model.layers[1:1]
i, ir=  10 8e-05
Using real-time data augmentation.
Epoch 1/1
1562/1562 [==============================] - 79s 50ms/step - loss: 0.1477 - acc: 0.9518 - val_loss: 0.4061 - val_acc: 0.8880

for layer in model.layers[1:10]
i, ir=  10 8e-05
Using real-time data augmentation.
Epoch 1/1
1562/1562 [==============================] - 69s 44ms/step - loss: 0.2200 - acc: 0.9266 - val_loss: 0.5886 - val_acc: 0.8288

AKB(32,32)
for layer in model.layers[1:1]:
i, ir=  190 1.4411518807585606e-06
Using real-time data augmentation.
Epoch 1/1
123/123 [==============================] - 6s 45ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 3.6459 - val_acc: 0.7389

AKB(64,64)
i, ir=  190 1.4411518807585606e-06
Using real-time data augmentation.
Epoch 1/1
123/123 [==============================] - 11s 88ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 2.8601 - val_acc: 0.7861

AKB(128,128)
i, ir=  190 1.4411518807585606e-06
Using real-time data augmentation.
Epoch 1/1
123/123 [==============================] - 23s 189ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 2.5846 - val_acc: 0.7833

AKB(224,224)
for layer in model.layers[1:15]:
i, ir=  190 1.4411518807585606e-06
Using real-time data augmentation.
Epoch 1/1
123/123 [==============================] - 27s 222ms/step - loss: 1.7911e-07 - acc: 1.0000 - val_loss: 1.8000 - val_acc: 0.8639
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
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
#from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

import numpy as np
import os
import shutil
import random
import matplotlib.pyplot as plt
#from keras.utils.visualize_util import plot


from getDataSet import getDataSet

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


batch_size = 32
num_classes = 10
epochs = 1
data_augmentation = False
img_rows=128
img_cols=128
result_dir="./history"

# The data, shuffled and split between train and test sets:
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train,y_train,x_test,y_test = getDataSet(img_rows,img_cols)
    #このままだと読み込んでもらえないので、array型にします。
    #x_train = np.array(x_train).astype(np.float32).reshape((len(x_train),3, 32, 32)) / 255
x_train = np.array(x_train)  #/ 255
y_train = np.array(y_train).astype(np.int32)
x_test = np.array(x_test) #/ 255
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
InceptionV3 = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC層を構築
top_model = Sequential()
top_model.add(Flatten(input_shape=InceptionV3.output_shape[1:])) #vgg16
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))

# VGG16とFCを接続
#model = Model(input=vgg16.input, output=top_model(vgg16.output))
#model = Model(input=vgg19.input, output=top_model(vgg19.output))
model = Model(input=InceptionV3.input, output=top_model(InceptionV3.output))

# 最後のconv層の直前までの層をfreeze
for layer in model.layers[1:10]:  #trainingするlayerを指定　15,10,1など
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

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

for i in range(epochs):
    epoch=100
    if not data_augmentation:
        print('Not using data augmentation.')
        """
        history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    nb_epoch=epoch,
                    verbose=1,
                    validation_split=0.1)
        """
        # 学習履歴をプロット
        #plot_history(history, result_dir)
        
        
        history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epoch,
                  validation_data=(x_test, y_test),
                  shuffle=True)
        
        # save weights every epoch
        model.save_weights('params_model_epoch_{0:03d}.hdf5'.format(i), True)   
        save_history(history, os.path.join(result_dir, 'history_epoch_{0:03d}.txt'.format(i)))
        
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
        history = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epoch,
                            validation_data=(x_test, y_test))
    if i%10==0:
        print('i, ir= ',i, lr)
        # save weights every epoch
        model.save_weights('params_model_VGG16L3_i_{0:03d}.hdf5'.format(i), True)
        """
        lr=lr*0.8
        opt = keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=1e-6)
        """
        # Let's train the model using Adam
        model.compile(loss='categorical_crossentropy',
                  optimizer=opt,metrics=['accuracy'])
    else:
        continue
        
save_history(history, os.path.join(result_dir, 'history.txt'))

        
