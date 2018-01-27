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

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import sys

import shutil
import random

"""
ImageNetで学習済みのVGG16モデルを使って入力画像のクラスを予測する
"""

if len(sys.argv) != 2:
    print("usage: python test_vgg16.py [image file]")
    sys.exit(1)

filename = sys.argv[1]
img_rows=224
img_cols=224
num_classes=10

# 学習済みのVGG16をロード
# 構造とともに学習済みの重みも読み込まれる
#model = VGG16(weights='imagenet')
# model.summary()

# VGG16モデルと学習済み重みをロード
# Fully-connected層（FC）はいらないのでinclude_top=False）
input_tensor = Input(shape=(img_rows, img_cols, 3))  #(img_rows, img_cols, 3)) x_train.shape[1:]
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC層を構築
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))

# VGG16とFCを接続
model = Model(input=vgg16.input, output=top_model(vgg16.output))
model.load_weights('params_model_VGG16L3_i_190.hdf5')
opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# モデルのサマリを表示
model.summary()


# 引数で指定した画像ファイルを読み込む
# サイズはVGG16のデフォルトである224x224にリサイズされる
img = image.load_img(filename, target_size=(img_rows,img_cols))

# 読み込んだPIL形式の画像をarrayに変換
#x = image.img_to_array(img)
x=np.array(img)/255
x = x.astype('float32')
# 3次元テンソル（rows, cols, channels) を
# 4次元テンソル (samples, rows, cols, channels) に変換
# 入力画像は1枚なのでsamples=1でよい
x = np.expand_dims(x, axis=0)

# Top-5のクラスを予測する
# VGG16の1000クラスはdecode_predictions()で文字列に変換される
#preds = model.predict(preprocess_input(x))
#print(x)
preds = model.predict(x)
results =preds #decode_predictions(preds, top=10)[0]
for result in results:
    print(result)
