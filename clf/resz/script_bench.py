# -*- coding: utf-8 -*-

from os import chdir, system
import sys

from datetime import datetime as dt

sys.path.insert(1, '../../aux')
from clfsettings import datasets, labels, shapes, train_datasets
from pathutils import jnt, lst, mkd

train_name = '_'.join(train_datasets)

time = dt.now().strftime('%Y_%b_%d_%H_%M_%S')
dst_base_dir = jnt('.', train_name + '__' + time)

#prepmode = '_preprocessed' # '_preprocessed' para usar as imagens pré-processadas ou '' caso contrário.  
prepmode = ''

for shape in shapes:    
    for dataset in datasets:
        for label in labels:
            dst_dir = jnt(dst_base_dir, shape, dataset, label); mkd(dst_dir) 
            chdir(dst_dir)                        
            print(f'=> Using the ImageMagick lossy method to resize the {label} images of the {dataset} dataset, as well as their correspondent segmentations and ground truths, to the target size {shape}x{shape}...')
            imgs_dir = jnt(dst_base_dir, '..', '..', '..', dataset + prepmode, label).replace(' ', '\ ')
            segs_dir = jnt(dst_base_dir, '..', '..', '..', 'seg', 'Postprocessing', 'dictK3_' + dataset.replace('-','').lower(), label).replace(' ', '\ ')
            system(f'convert {imgs_dir}/*.jpg -resize {shape}x{shape}^ -gravity center -extent {shape}x{shape} -set filename:base \"%[basename]\" \"%[filename:base].jpg\"')
            system(f'convert {segs_dir}/*.png -resize {shape}x{shape}^ -gravity center -extent {shape}x{shape} -set filename:base \"%[basename]\" \"%[filename:base]_seg.png\"')
            if dataset in ['DermIS', 'DermNet', 'DermQuest']: # datasets que possuem ground truths.
                system(f'convert {imgs_dir}/*.png -resize {shape}x{shape}^ -gravity center -extent {shape}x{shape} -set filename:base \"%[basename]\" \"%[filename:base]_gt.png\"')
            chdir(jnt(dst_base_dir, '..'))
print('=> Done!')
