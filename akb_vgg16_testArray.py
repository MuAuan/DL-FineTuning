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
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import sys

import shutil
import random
import cv2
import os

"""
ImageNetで学習済みのVGG16モデルを使って入力画像のクラスを予測する
"""
"""
if len(sys.argv) != 2:
    print("usage: python test_vgg16.py [image file]")
    sys.exit(1)
"""
#filename = sys.argv[1]
img_rows=224
img_cols=224
num_classes=18

# 学習済みのVGG16をロード
# 構造とともに学習済みの重みも読み込まれる
#model = VGG16(weights='imagenet')
# model.summary()

# VGG16モデルと学習済み重みをロード
# Fully-connected層（FC）はいらないのでinclude_top=False）
input_tensor = Input(shape=(img_rows, img_cols, 3))  #(img_rows, img_cols, 3)) x_train.shape[1:]
#vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
ResNet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC層を構築
top_model = Sequential()
top_model.add(Flatten(input_shape=ResNet50.output_shape[1:])) #VGG16
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))

# VGG16とFCを接続
#model = Model(input=vgg16.input, output=top_model(vgg16.output))
model = Model(input=ResNet50.input, output=top_model(ResNet50.output))
model.load_weights('params_model_epoch_000.hdf5')
lr = 0.00001
opt = keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=1e-6)
#opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9)

for layer in model.layers[1:90]:  
    layer.trainable = False
3  
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# モデルのサマリを表示
model.summary()


# 引数で指定した画像ファイルを読み込む
# サイズはVGG16のデフォルトである224x224にリサイズされる
#ここでは学習時の128×128
#リストの作成
X_train = []
X_test = []
y_train = []
y_test = []

path = "./train_images"
i=6
cutNum = 60 #700
cutNum2 = 48 #630
imgList = os.listdir(path+str(i))
print(imgList)
imgNum = len(imgList)
for j in range(cutNum):
    #imgSrc = cv2.imread(path+str(i)+"/"+imgList[j])
    #target_size=(img_rows,img_cols)のサイズのimage
    img = image.load_img(path+str(i)+"/"+imgList[j], target_size=(img_rows,img_cols))
    imgSrc = image.img_to_array(img)
                        
    if imgSrc is None:continue
    if j < cutNum2:
        X_train.append(imgSrc)
        y_train.append(i)
    else:
        X_test.append(imgSrc)
        y_test.append(i)
print(len(X_train),len(y_train),len(X_test),len(y_test))

x_train = np.array(X_train)  / 255
y_train = np.array(y_train).astype(np.int32)
x_test = np.array(X_test) / 255
y_test = np.array(y_test).astype(np.int32)

# Convert class vectors to binary class matrices.
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
#img = image.load_img(filename, target_size=(img_rows,img_cols))

# 読み込んだPIL形式の画像をarrayに変換
#x = image.img_to_array(img)
#x=np.array(img)/255
#x = x.astype('float32')
# 3次元テンソル（rows, cols, channels) を
# 4次元テンソル (samples, rows, cols, channels) に変換
# 入力画像は1枚なのでsamples=1でよい
x_train = np.expand_dims(x_train, axis=0)
x_test = np.expand_dims(x_test, axis=0)

# Top-5のクラスを予測する
# VGG16の1000クラスはdecode_predictions()で文字列に変換される
#preds = model.predict(preprocess_input(x))
#print(x_train[0])

preds = model.predict(x_train[0])
#preds = model.predict(x_test[0])
results =preds #decode_predictions(preds, top=10)[0]
j=0
for result in results:
    print(result.argmax()) #,result.max())
    j += 1
preds = model.predict(x_test[0])
results =preds #decode_predictions(preds, top=10)[0]
for result in results:
    print(result.argmax()) #  ,result.max())
    j += 1
