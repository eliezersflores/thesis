#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from datetime import datetime as dt
import numpy as np
import scipy.io as sio
import tensorflow as tf
from time import time as get_time
sys.path.insert(1, '../../aux')

from clfsettings import cnns, train_datasets
from pathutils import jnt, lst, mkd

train_name = '_'.join(train_datasets)

masks_base_dir = jnt('..', 'masks', train_name, checkparts=True)

time = dt.now().strftime('%Y_%b_%d_%H_%M_%S')
dst_base_dir = jnt('.', train_name + '__' + time)
mkd(dst_base_dir)

for cnn in reversed(cnns):

    ti = get_time()

    if cnn == 'densenet121':
        from tensorflow.keras.applications import DenseNet121 as convnet
        from tensorflow.keras.applications.densenet import preprocess_input
        img_side = 224
    elif cnn == 'densenet169':
        from tensorflow.keras.applications import DenseNet169 as convnet
        from tensorflow.keras.applications.densenet import preprocess_input
        img_side = 224
    elif cnn == 'densenet201':
        from tensorflow.keras.applications import DenseNet201 as convnet
        from tensorflow.keras.applications.densenet import preprocess_input
        img_side = 224
    elif cnn == 'efficientnetb0':
        from tensorflow.keras.applications import EfficientNetB0 as convnet
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_side = 224
    elif cnn == 'efficientnetb1':
        from tensorflow.keras.applications import EfficientNetB1 as convnet
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_side = 240
    elif cnn == 'efficientnetb2':
        from tensorflow.keras.applications import EfficientNetB2 as convnet
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_side = 260
    elif cnn == 'efficientnetb3':
        from tensorflow.keras.applications import EfficientNetB3 as convnet
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_side = 300
    elif cnn == 'efficientnetb4':
        from tensorflow.keras.applications import EfficientNetB4 as convnet
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_side = 380
    elif cnn == 'efficientnetb5':
        from tensorflow.keras.applications import EfficientNetB5 as convnet
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_side = 456
    elif cnn == 'efficientnetb6':
        from tensorflow.keras.applications import EfficientNetB6 as convnet
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_side = 528
    elif cnn == 'efficientnetb7':
        from tensorflow.keras.applications import EfficientNetB7 as convnet
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_side = 600
    elif cnn == 'inceptionresnetv2':
        from tensorflow.keras.applications import InceptionResNetV2 as convnet
        from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
        img_side = 299
    elif cnn == 'inceptionv3':
        from tensorflow.keras.applications import InceptionV3 as convnet
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        img_side = 299
    elif cnn == 'mobilenet':
        from tensorflow.keras.applications import MobileNet as convnet
        from tensorflow.keras.applications.mobilenet import preprocess_input
        img_side = 224
    elif cnn == 'mobilenetv2':
        from tensorflow.keras.applications import MobileNetV2 as convnet
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img_side = 224
    elif cnn == 'mobilenetv3large':
        from tensorflow.keras.applications import MobileNetV3Large as convnet
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
        img_side = 224
    elif cnn == 'mobilenetv3small':
        from tensorflow.keras.applications import MobileNetV3Small as convnet
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
        img_side = 224
    elif cnn == 'nasnetlarge':
        from tensorflow.keras.applications import NASNetLarge as convnet
        from tensorflow.keras.applications.nasnet import preprocess_input
        img_side = 331
    elif cnn == 'nasnetmobile':
        from tensorflow.keras.applications import NASNetMobile as convnet
        from tensorflow.keras.applications.nasnet import preprocess_input
        img_side = 224
    elif cnn == 'resnet101':
        from tensorflow.keras.applications import ResNet101 as convnet
        from tensorflow.keras.applications.resnet import preprocess_input
        img_side = 224
    elif cnn == 'resnet101v2':
        from tensorflow.keras.applications import ResNet101V2 as convnet
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
        img_side = 224
    elif cnn == 'resnet152':
        from tensorflow.keras.applications import ResNet152 as convnet
        from tensorflow.keras.applications.resnet import preprocess_input
        img_side = 224
    elif cnn == 'resnet152v2':
        from tensorflow.keras.applications import ResNet152V2 as convnet
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
        img_side = 224
    elif cnn == 'resnet50':
        from tensorflow.keras.applications import ResNet50 as convnet
        from tensorflow.keras.applications.resnet import preprocess_input
        img_side = 224
    elif cnn == 'resnet50v2':
        from tensorflow.keras.applications import ResNet50V2 as convnet
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
        img_side = 224
    elif cnn == 'vgg16':
        from tensorflow.keras.applications import VGG16 as convnet
        from tensorflow.keras.applications.vgg16 import preprocess_input
        img_side = 224
    elif cnn == 'vgg19':
        from tensorflow.keras.applications import VGG19 as convnet
        from tensorflow.keras.applications.vgg19 import preprocess_input
        img_side = 224
    elif cnn == 'xception':
        from tensorflow.keras.applications import Xception as convnet
        from tensorflow.keras.applications.xception import preprocess_input
        img_side = 299

    model = convnet(include_top=False, weights='imagenet', pooling='avg')
    src_base_dir = jnt(masks_base_dir, str(img_side))

    for fold in lst(src_base_dir):

        src_dir = jnt(src_base_dir, fold)

        dst_dir = jnt(dst_base_dir, cnn, fold)
        mkd(dst_dir)

        # Para evitar extrair novamente features que já foram extraídas anteriormente, em uma eventual execução interrompida de maneira inesperada. 
        if os.path.exists(jnt(dst_dir, 'data.mat')):
            continue

        print('=> Performing feature extraction by using {:s} convnet over the {:s} fold...'.format(cnn, fold))

        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            src_dir, 
            labels='inferred', 
            label_mode='binary',
            class_names=['notmelanoma','melanoma'],
            color_mode='rgb',
            batch_size=32,
            image_size=(img_side,img_side),
            shuffle=False,
            validation_split=0,
        )
        batch_idx = 0
        for batch_images, batch_labels in dataset:
            batch_imges_pp = preprocess_input(batch_images)
            if batch_idx == 0:
                features = model.predict(batch_imges_pp)
                labels = batch_labels
            else:
                features = np.vstack([features, model.predict(batch_imges_pp)])
                labels = np.vstack([labels, batch_labels])
            batch_idx += 1

        dt = get_time() - ti

        sio.savemat(jnt(dst_dir, 'data.mat'), {'features':features, 'labels':labels})
        np.savetxt(jnt(dst_dir, 'elapsed_time.csv'), [dt], delimiter=',') 
