import sys

import csv
from datetime import datetime as dt
from numpy import argmin, argmax, median, squeeze, sqrt
import scipy.io as sio
from sklearn.metrics import auc, confusion_matrix, roc_curve, roc_auc_score

sys.path.insert(1, '../../aux')
from clfsettings import cnns, datasets, lambs, n_folds, train_datasets, vphis
from ddf import DDF
from pathutils import jnt, lst, mkd
from vlfeat.wrapper import gmm

train_name = '_'.join(train_datasets)

folds_base_dir = jnt('..', 'folds', train_name, checkparts=True)
feats_base_dir = jnt('..', 'feats', train_name, checkparts=True)

time = dt.now().strftime('%Y_%b_%d_%H_%M_%S')
dst_dir = jnt('.', train_name + '__' + time); mkd(dst_dir)

cnn = 'resnet50'
lamb = 5 
vphi = 0.8 

ftrain_contents = sio.loadmat(jnt(feats_base_dir, cnn, train_name, 'data.mat'))
Yftrain = ftrain_contents['features'].T
Tftrain = squeeze(ftrain_contents['labels'])
    
clf = DDF(lamb, vphi)
clf.fit(Yftrain, Tftrain)

for dataset in datasets:

    test_contents = sio.loadmat(jnt(feats_base_dir, cnn, dataset, 'data.mat'))
    Ytest = test_contents['features'].T
    Ttest = squeeze(test_contents['labels'])

    pred_proba = clf.predict_proba(Ytest)
    #pred = argmax(pred_proba, axis=0)
   
    fpr, tpr, thresholds = roc_curve(Ttest, pred_proba[1,:])
    rauc = auc(fpr, tpr)
    th = thresholds[argmin(abs(tpr - (1 - fpr)))]

    pred = (pred_proba[1,:] > th).astype('float')
    C = confusion_matrix(Ttest, pred)
    print(C)
    TN = C[0, 0]
    FP = C[0, 1]
    FN = C[1, 0]
    TP = C[1, 1]
    sens = TP/(TP + FN)
    spec = TN/(TN + FP)
    bacc = (sens + spec)/2
    gmean = sqrt(sens*spec)

    with    open(jnt(folds_base_dir, dataset + '.csv'), 'r') \
            as src_file, \
            open(jnt(dst_dir, f'{dataset}_{cnn}_{lamb}_{vphi}.csv'), 'w') \
            as dst_file:

            reader = csv.reader(src_file)
            writer = csv.writer(dst_file)
            writer.writerow(['dataset', 'label', 'img', 'melscore'])
            for row_idx, row in enumerate(reader):
                row.append(pred_proba[1, row_idx])
                writer.writerow(row)

    print('. {:9s} => sens = {:6.2f}, spec = {:6.2f}, rauc = {:6.2f}, bacc = {:6.2f}, gmean = {:6.2f}'.format(dataset, 100*sens, 100*spec, 100*rauc, 100*bacc, 100*gmean))
