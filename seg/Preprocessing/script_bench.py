#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
from PIL import Image
from time import time

from shad_atten import shad_atten as shad_atten

# EDIT HERE (1 of 2):
verbose = 1

# EDIT HERE (2 of 2):
#dataset_name = 'DermNet'
#dataset_name = 'DermIS'
#dataset_name = 'DermQuest'
#dataset_name = 'MED-NODE'
#dataset_name = 'PAD-UFES'
#dataset_name = 'MClass-ND'
dataset_name = 'MClass-D'

# Setting the source and destination dirs:
src_dir_root = os.path.join('/home/eliezer/Área de Trabalho/thesis', dataset_name)
src_dir_melanoma = os.path.join(src_dir_root, 'melanoma')
src_dir_notmelanoma = os.path.join(src_dir_root, 'notmelanoma')

# Creating the destination dirs, if they do not exist:
curr_dir = os.getcwd()
if verbose == 0:  
    dst_dir_root = os.path.join('/home/eliezer/Área de Trabalho/thesis', dataset_name + '_preprocessed')
else:
    dst_dir_root = os.path.join('/home/eliezer/Área de Trabalho/thesis/seg/Preprocessing', dataset_name + '_preprocessed')
dst_dir_melanoma = os.path.join(dst_dir_root, 'melanoma')
dst_dir_notmelanoma = os.path.join(dst_dir_root, 'notmelanoma')
if not os.path.exists(dst_dir_root):
    os.mkdir(dst_dir_root)
    os.mkdir(dst_dir_melanoma)
    os.mkdir(dst_dir_notmelanoma)

elapsed_times = []

# Preprocessing all the images of the selected dataset.
for src_dir, dst_dir in zip([src_dir_melanoma, src_dir_notmelanoma], [dst_dir_melanoma, dst_dir_notmelanoma]):
    
    fnames_tmp = os.listdir(src_dir)
    fnames = [fname for fname in fnames_tmp if fname.endswith('.jpg')]    
    nimgs = len(fnames)
    
    for i in range(nimgs):
    
        print('Working on {:s} image {:3d} of {:3d}...'.format(dst_dir.split('/')[-1], i + 1, nimgs))
        
        name = fnames[i].split('.')[0]

        img_tmp = cv2.imread(os.path.join(src_dir, name + '.jpg'))
        img = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)
        if dataset_name in ['DermNet','DermIS','DermQuest']:
            gt_tmp = cv2.imread(os.path.join(src_dir, name + '.png'))
            gt = cv2.cvtColor(gt_tmp, cv2.COLOR_BGR2GRAY)
            
        ti = time()
                
        nr = img.shape[0]
        nc = img.shape[1]
        
        kappa = np.int(0.2*min(nr,nc))
        mask = np.zeros((nr, nc), dtype=int)
        mask[0:kappa, 0:kappa] = 1
        mask[0:kappa, nc-kappa:nc] = 1
        mask[nr-kappa:nr, 0:kappa] = 1
        mask[nr-kappa:nr, nc-kappa:nc] = 1
                
        saturate_opt = 1
        normalize_opt = 1    
        
        if verbose == 0:
        
            img_corr, _, _, _ = shad_atten(img, mask, saturate_opt, normalize_opt)    
            
            img_pil = Image.fromarray(img_corr, 'RGB')
            
            dt = time() - ti
            elapsed_times.append(dt)
            
            img_pil.save(os.path.join(dst_dir, name + '.jpg'))
            
            if dataset_name in ['DermNet','DermIS','DermQuest']:
                gt_pil = Image.fromarray(gt, 'L')
                gt_pil.save(os.path.join(dst_dir, name + '.png'))            
            
        else:
        
            img_pil = Image.fromarray(img, 'RGB')
            img_pil.save(os.path.join(dst_dir, name + '_orig.jpg'))        
        
            img_corr, Vin, Vout, Z = shad_atten(img, mask, saturate_opt, normalize_opt)
        
            imgcorr_pil = Image.fromarray(img_corr, 'RGB')
            imgcorr_pil.save(os.path.join(dst_dir, name + '_corr.jpg'))
    
            Vin_pil = Image.fromarray(Vin, 'L')
            Vin_pil.save(os.path.join(dst_dir, name + '_Vorig.jpg'))
            
            Vout_pil = Image.fromarray(Vout, 'L')
            Vout_pil.save(os.path.join(dst_dir, name + '_Vcorr.jpg'))
            
            Z_pil = Image.fromarray(Z, 'L')
            Z_pil.save(os.path.join(dst_dir, name + '_Z.jpg'))
     
np.savetxt(os.path.join(dst_dir_root, 'elapsed_time.csv'), elapsed_times, delimiter=',')            
