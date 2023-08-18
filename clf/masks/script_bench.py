# -*- coding: utf-8 -*-

from re import search
import sys

from csv import reader
import numpy as np

from datetime import datetime as dt

sys.path.insert(1, '../../aux')
from clfsettings import datasets, train_datasets, shapes
from imgutils import get_img, save_img
from pathutils import jnt, lst

train_name = '_'.join(train_datasets)

bkgds_base_dir = jnt('..', 'bkgds', train_name, checkparts=True)
folds_base_dir = jnt('..', 'folds', train_name, checkparts=True)
reszs_base_dir = jnt('..', 'reszs', train_name, checkparts=True)

csvfnames = lst(folds_base_dir, fmt='csv', inc_fmt=False)

time = dt.now().strftime('%Y_%b_%d_%H_%M_%S')
dst_base_dir = jnt('.', train_name + '__' + time)

for shape in shapes:
    for csvfname in csvfnames:
        print('=> \"Removing\" the background of the images previously resized to {:s}x{:s} and listed in the {:s}.csv file...'.format(shape, shape, csvfname))
        with open(jnt(folds_base_dir, csvfname + '.csv'), mode='r') as csvfile:
            csv = reader(csvfile)
            for row in csv:

                dataset, label, name = row
                
                img = get_img(jnt(reszs_base_dir, shape, dataset, label, name + '.jpg'))

                seg = get_img(jnt(reszs_base_dir, shape, dataset, label, name + '_seg.png'))

                seg[np.where(seg <= 127)] = 0
                seg[np.where(seg > 127)] = 255
 
                # p1 é a porcentagem de pixels segmentados como lesão.
                s0 = sum(sum(seg == 0))
                s1 = sum(sum(seg == 255))
                p1 = s1/(s0 + s1)
                
                # q é a quantidade de pixels segmentados como lesão e dentro do quadrado central que contém um quarto da dimensão original da imagem. 
                #d = int(float(shape)/4)
                d = int(3*float(shape)/8)
                Q = seg[d+1:-d+1,d+1:-d+1]
                q = sum(sum(Q!=0))

                if p1 < 1.5/100 or q == 0:
                    seg[np.where(seg == 0)] = 255 # coloca tudo como foreground.

                if csvfname.startswith('train') or csvfname.startswith('valid'):
                    fold = search('\D(\d+)', csvfname)[1]
                    bkg = get_img(jnt(bkgds_base_dir, shape, 'train' + fold + '.jpg'))
                else:
                    bkg = get_img(jnt(bkgds_base_dir, shape, 'all.jpg'))

                new_img = img
                new_img[np.where(seg == 0)] = bkg[np.where(seg == 0)]

                save_img(new_img, jnt(dst_base_dir, shape, csvfname, label, dataset + '_' + name + '.jpg'))
