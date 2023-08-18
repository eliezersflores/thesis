#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
from PIL import Image

# EDIT HERE (1 of 5):
#src_dir_melanoma = '/home/eliezer/Dropbox/Datasets/Skin Image Data Set-1/skin_data/melanoma/dermIS'
src_dir_melanoma = '/home/eliezer/Dropbox/Datasets/Skin Image Data Set-1/skin_data/melanoma/dermquest'
# EDIT HERE (2 of 5):
#src_dir_notmelanoma = '/home/eliezer/Dropbox/Datasets/Skin Image Data Set-2/skin_data/notmelanoma/dermis'
src_dir_notmelanoma = '/home/eliezer/Dropbox/Datasets/Skin Image Data Set-2/skin_data/notmelanoma/dermquest'

# EDIT HERE (3 of 5):
#dst_dir_root = '/home/eliezer/Área de Trabalho/Experimentos Tese/DermIS/'
dst_dir_root = '/home/eliezer/Área de Trabalho/Experimentos Tese/DermQuest/'
# EDIT HERE (4 of 5):
#dst_dir_melanoma = '/home/eliezer/Área de Trabalho/Experimentos Tese/DermIS/melanoma'
dst_dir_melanoma = '/home/eliezer/Área de Trabalho/Experimentos Tese/DermQuest/melanoma'
# EDIT HERE (5 of 5):
#dst_dir_notmelanoma = '/home/eliezer/Área de Trabalho/Experimentos Tese/DermIS/notmelanoma'
dst_dir_notmelanoma = '/home/eliezer/Área de Trabalho/Experimentos Tese/DermQuest/notmelanoma'

# Creating the destination dirs, if they do not exist:
if not os.path.exists(dst_dir_root):
    os.makedirs(dst_dir_root)
    os.makedirs(dst_dir_melanoma)
    os.makedirs(dst_dir_notmelanoma)

# Preparing all the images of the selected dataset.
for src_dir, dst_dir in zip([src_dir_melanoma, src_dir_notmelanoma], [dst_dir_melanoma, dst_dir_notmelanoma]):
    
    fnames_tmp = os.listdir(src_dir)
    fnames = [fname for fname in fnames_tmp if fname.endswith('.jpg')]    
    nimgs = len(fnames)

    for i in range(nimgs):
        
        print('Working on {:s} image {:3d} of {:3d}...'.format(dst_dir.split('/')[-1], i + 1, nimgs))
        
        name = '_'.join(fnames[i].split('_')[:-1])
                
        img_tmp = cv2.imread(os.path.join(src_dir, name + '_orig.jpg'))
        img = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)
        gt_tmp = cv2.imread(os.path.join(src_dir, name + '_contour.png'))
        gt = cv2.cvtColor(gt_tmp, cv2.COLOR_BGR2GRAY)    
        
        img_pil = Image.fromarray(img, 'RGB')
        img_pil.save(os.path.join(dst_dir, name + '.jpg'))
        
        gt_pil = Image.fromarray(gt, 'L')
        gt_pil.save(os.path.join(dst_dir, name + '.png'))        
