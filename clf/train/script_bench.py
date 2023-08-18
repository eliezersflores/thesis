import sys

import csv
from datetime import datetime as dt
from numpy import argmin, argmax, median, squeeze, sqrt
import scipy.io as sio
from sklearn.metrics import auc, confusion_matrix, roc_curve

sys.path.insert(1, '../../aux')
from clfsettings import cnns, datasets, lambs, n_folds, train_datasets, vphis
from ddf import DDF
from pathutils import jnt, lst, mkd
from vlfeat.wrapper import gmm

train_name = '_'.join(train_datasets)
#feats_base_dir = jnt('..', 'feats', train_name, checkparts=True)
feats_base_dir = jnt('..', 'feats', 'DermIS_DermQuest__2021_Nov_22_14_57_59')

time = dt.now().strftime('%Y_%b_%d_%H_%M_%S')
dst_base_dir = jnt('.', train_name + '__' + time)
mkd(dst_base_dir)

with open(jnt(dst_base_dir, f'results_{train_name}.csv'), 'w') as csvfile:
    csvwriter = csv.writer(csvfile)

    [max_rauc, opt_rauc_cnn, opt_rauc_lamb, opt_rauc_vphi] = [0, '', 0, 0]
    [max_bacc, opt_bacc_cnn, opt_bacc_lamb, opt_bacc_vphi] = [0, '', 0, 0]
    [max_gmean, opt_gmean_cnn, opt_gmean_lamb, opt_gmean_vphi] = [0, '', 0, 0]

    for cnn in cnns:

        for lamb in lambs:

            for vphi in vphis:

                print('* cnn = {:s}, lamb = {:.2f} and vphi = {:.1f}'.format(cnn, lamb, vphi))
                method = '_'.join([cnn, str(lamb), str(vphi)])

                raucs = []
                baccs = []
                gmeans = []

                for fold in range(1, n_folds + 1):

                    train_contents = sio.loadmat(jnt(feats_base_dir, cnn, 'train' + str(fold), 'data.mat'))
                    Ytrain = train_contents['features'].T
                    Ttrain = squeeze(train_contents['labels'])

                    valid_contents = sio.loadmat(jnt(feats_base_dir, cnn, 'valid' + str(fold), 'data.mat'))
                    Yvalid = valid_contents['features'].T
                    Tvalid = squeeze(valid_contents['labels'])

                    clf = DDF(lamb, vphi)
                    clf.fit(Ytrain, Ttrain)

                    pred_proba = clf.predict_proba(Yvalid)
                   
                    fpr, tpr, thresholds = roc_curve(Tvalid, pred_proba[1,:])
                    rauc = auc(fpr, tpr)
                    th = thresholds[argmin(abs(tpr - (1 - fpr)))]

                    pred = (pred_proba[1,:] > th).astype('float')
                    C = confusion_matrix(Tvalid, pred)
                    TN = C[0, 0]
                    FP = C[0, 1]
                    FN = C[1, 0]
                    TP = C[1, 1]
                    sens = TP/(TP + FN)
                    spec = TN/(TN + FP)
                    bacc = (sens + spec)/2
                    gmean = sqrt(sens*spec)

                    print('. fold = {:2d} => rauc = {:5.2f}, bacc = {:5.2f}, gmean = {:5.2f}'.format(fold, 100*rauc, 100*bacc, 100*gmean))
 
                    raucs.append(rauc)
                    baccs.append(bacc)
                    gmeans.append(gmean)

                csvwriter.writerow([method, *raucs, *baccs, *gmeans])

"""

                auc = median(aucs)
                bacc = median(baccs)
                gmean = median(gmean)


                if auc > max_auc:
                    opt_auc_cnn = cnn
                    opt_auc_lamb = lamb
                    opt_auc_vphi = vphi
                    max_auc = auc 

                if bacc > max_bacc:
                    opt_bacc_cnn = cnn
                    opt_bacc_lamb = lamb
                    opt_bacc_vphi = vphi
                    max_bacc = bacc

                if gmean > max_gmean:
                    opt_gmean_cnn = cnn
                    opt_gmean_lamb = lamb
                    opt_gmean_vphi = vphi
                    max_bacc = bacc

    print('* opt_auc_cnn = {:s}, opt_auc_lamb = {:.3f} and opt_auc_vphi = {:.1f} => valid auc = {:.2f}'.format(opt_auc_cnn, opt_auc_lamb, opt_auc_vphi, 100*max_auc))
    f.write('* opt_auc_cnn = {:s}, opt_auc_lamb = {:.3f} and opt_auc_vphi = {:.1f} => valid auc = {:.2f}\n'.format(opt_auc_cnn, opt_auc_lamb, opt_auc_vphi, 100*max_auc))

    print('* opt_bacc_cnn = {:s}, opt_bacc_lamb = {:.3f} and opt_bacc_vphi = {:.1f} => valid bacc = {:.2f}'.format(opt_bacc_cnn, opt_bacc_lamb, opt_bacc_vphi, 100*max_bacc))
    f.write('* opt_bacc_cnn = {:s}, opt_bacc_lamb = {:.3f} and opt_bacc_vphi = {:.1f} => valid bacc = {:.2f}\n'.format(opt_bacc_cnn, opt_bacc_lamb, opt_bacc_vphi, 100*max_bacc))

    print('* opt_gmean_cnn = {:s}, opt_gmean_lamb = {:.3f} and opt_gmean_vphi = {:.1f} => valid gmean = {:.2f}'.format(opt_gmean_cnn, opt_gmean_lamb, opt_gmean_vphi, 100*max_gmean))
    f.write('* opt_gmean_cnn = {:s}, opt_gmean_lamb = {:.3f} and opt_gmean_vphi = {:.1f} => valid gmean = {:.2f}\n'.format(opt_gmean_cnn, opt_gmean_lamb, opt_gmean_vphi, 100*max_gmean))

"""

"""

    # Testes:

    ftrain_contents = sio.loadmat(jnt(feats_base_dir, opt_cnn, train_name, 'data.mat'))
    Yftrain = ftrain_contents['features'].T
    Tftrain = squeeze(ftrain_contents['labels'])
        
    clf = DDF(opt_lamb, opt_vphi)
    clf.fit(Yftrain, Tftrain)

    for test_dataset in datasets:

        test_contents = sio.loadmat(jnt(feats_base_dir, opt_cnn, test_dataset, 'data.mat'))
        Ytest = test_contents['features'].T
        Ttest = squeeze(test_contents['labels'])

        pred = clf.predict(Ytest)

        bacc = balanced_accuracy_score(Ttest, pred)
        print('=> opt_lamb = {:.3f} and opt_vphi = {:.1f} => {} bacc = {:.2f}'.format(opt_lamb, opt_vphi, test_dataset, 100*bacc))
        f.write('=> opt_lamb = {:.3f} and opt_vphi = {:.1f} => {} bacc = {:.2f}\n'.format(opt_lamb, opt_vphi, test_dataset, 100*bacc))

    del clf

"""
