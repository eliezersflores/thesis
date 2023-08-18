import sys

from numpy import log2
from numpy.random import seed
from pandas import DataFrame
from scipy.io import loadmat
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

sys.path.insert(1, '../../../aux')
from clfsettings import train_datasets
from pathutils import jnt
from plotutils import plotKtuning

train_name = '_'.join(train_datasets)
feats_base_dir = jnt('../..', 'feats', train_name, 'resnet50', checkparts=True)

seed(5489)

C_set = [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]
perfs_df = DataFrame()

for fold_idx in range(1, 11):

    for C in C_set :

        print(f'Training and validating with fold_idx = {fold_idx} and C = {C}')

        train_fpath = jnt(feats_base_dir, 'train' + str(fold_idx), 'data.mat')
        train_contents = loadmat(train_fpath)
        Ytrain = train_contents['features']
        Ttrain = train_contents['labels']

        valid_fpath = jnt(feats_base_dir, 'valid' + str(fold_idx), 'data.mat')
        valid_contents = loadmat(valid_fpath)
        Yvalid = valid_contents['features']
        Tvalid = valid_contents['labels'] 
 
        clf = SVC(kernel='linear', C=C, probability=True)
        clf.fit(Ytrain, Ttrain.ravel())
        
        preds = clf.predict_proba(Yvalid)
        perfs_df.loc[fold_idx, C] = roc_auc_score(Tvalid, preds[:,1])

print('Saving plot...')

xlabel = '$\\log_{{2}}({{C}})$'
xticklabels = ['{:.0f}'.format(log2(e)) for e in perfs_df.columns.to_list()]


df = plotKtuning(perfs_df, imgname='tuning_SVM.pdf', linewidth_frac=0.45, xlabel=xlabel, xticklabels=xticklabels, xticklabels_size=7.5, yticklabels_size=7.5)

print('Done!')
