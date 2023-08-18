#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 19:37:05 2020

@author: eliezer
"""

import cv2
import numpy as np
import os
from PIL import Image
from time import time
import sys

from tensorflow.keras import models

curr_dir = os.getcwd()     
sys.path.append(os.path.join(curr_dir, 'src'))

# Loading convnet and its weights:
#height=192
#width=256
#convnet = DSNet(2, height, width)

json_file = open('ftu-convnet.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

convnet = models.model_from_json(loaded_model_json)

convnet.load_weights('weights.hdf5')

sys.exit()

# EDIT HERE (1 of 1):
dataset_name = 'DermNet'
#dataset_name = 'DermQuest'

# Setting the source and destination dirs:
src_dir_root = os.path.join('/home/eliezer/Área de Trabalho/Experimentos Tese', dataset_name)
src_dir_melanoma = os.path.join(src_dir_root, 'melanoma')
src_dir_notmelanoma = os.path.join(src_dir_root, 'notmelanoma')

# Creating the destination dirs, if they do not exist:
curr_dir = os.getcwd()
dst_dir_root = os.path.join('/home/eliezer/Área de Trabalho/Experimentos Tese/Related Works/DSNET', dataset_name + '_results')
dst_dir_melanoma = os.path.join(dst_dir_root, 'melanoma')
dst_dir_notmelanoma = os.path.join(dst_dir_root, 'notmelanoma')
if not os.path.exists(dst_dir_root):
    os.mkdir(dst_dir_root)
    os.mkdir(dst_dir_melanoma)
    os.mkdir(dst_dir_notmelanoma)

elapsed_times = []

# Segmenting all the images of the selected dataset.
for src_dir, dst_dir in zip([src_dir_melanoma, src_dir_notmelanoma], [dst_dir_melanoma, dst_dir_notmelanoma]):
    
    fnames_tmp = os.listdir(src_dir)
    fnames = [fname for fname in fnames_tmp if fname.endswith('.jpg')]    
    nimgs = len(fnames)
    
    for i in range(nimgs):
    
        print('Segmenting {:s} image {:3d} of {:3d}...'.format(dst_dir.split('/')[-1], i + 1, nimgs))
        
        name = fnames[i].split('.')[0]

        img_tmp = cv2.imread(os.path.join(src_dir, name + '.jpg'))
        img = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)
            
        ti = time()
    
        img_rows = img.shape[0]
        img_cols = img.shape[1]
           
        img = img/255
                
        img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        img_reshaped = np.expand_dims(img_resized, axis=0)
            
        seg_reshaped = convnet.predict(img_reshaped)
            
        seg_resized = seg_reshaped[0,:,:,0]
        seg_resized[seg_resized > 0.5] = 1
        seg_resized[seg_resized <= 0.5] = 0
        
        seg = cv2.resize(seg_resized, (img_cols, img_rows), interpolation=cv2.INTER_LINEAR)
     
        dt = time() - ti
        elapsed_times.append(dt)
        
        img_pil = Image.fromarray(np.uint8(255*seg), 'L')
        img_pil.save(os.path.join(dst_dir, name + '.png'))
     
np.savetxt(os.path.join(dst_dir_root, 'elapsed_time.csv'), elapsed_times, delimiter=',')
