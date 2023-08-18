#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
from PIL import Image

# EDIT HERE (1 of 6):
src_dir_melanoma = '/home/eliezer/Dropbox/Datasets/msl_dermnet/melanomas'
# EDIT HERE (2 of 6):
src_dir_notmelanoma = '/home/eliezer/Dropbox/Datasets/msl_dermnet/nevus'
# EDIT HERE (3 of 6):
src_dir_gts = '/home/eliezer/Dropbox/Datasets/msl_dermnet/gts'

# EDIT HERE (4 of 6):
dst_dir_root = '/home/eliezer/Área de Trabalho/Experimentos Tese/DermNet/'
# EDIT HERE (5 of 6):
dst_dir_melanoma = '/home/eliezer/Área de Trabalho/Experimentos Tese/DermNet/melanoma'
# EDIT HERE (6 of 6):
dst_dir_notmelanoma = '/home/eliezer/Área de Trabalho/Experimentos Tese/DermNet/notmelanoma'

# Creating the destination dirS, if they do not exist:
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
        
        name = fnames[i].split('.')[0]
        
        img_tmp = cv2.imread(os.path.join(src_dir, name + '.jpg'))
        img = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)
        gt_tmp = cv2.imread(os.path.join(src_dir_gts, name + '.tif'))
        gt = cv2.cvtColor(gt_tmp, cv2.COLOR_BGR2GRAY)    
        
        img_rows = img.shape[0]
        img_cols = img.shape[1]
        
        gt_rows = gt.shape[0]
        gt_cols = gt.shape[1]
        
        if img_rows != gt_rows or img_cols != gt_cols:
            img = cv2.resize(img, (gt_cols, gt_rows), cv2.INTER_NEAREST)
            
        img_pil = Image.fromarray(img, 'RGB')
        img_pil.save(os.path.join(dst_dir, name + '.jpg'))
        
        gt_pil = Image.fromarray(gt, 'L')
        gt_pil.save(os.path.join(dst_dir, name + '.png'))        