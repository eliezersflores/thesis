import sys

from numpy.random import seed
from pandas import DataFrame
from scipy.io import loadmat
import seaborn as sns
from sklearn.metrics import roc_auc_score

sys.path.insert(1, '../../../aux')
from clfsettings import train_datasets
from ddf import DDF
from pathutils import jnt
from plotutils import plotKtuning
from vlfeat.wrapper import gmm

train_name = '_'.join(train_datasets)
feats_base_dir = jnt('../..', 'feats', train_name, 'resnet50', checkparts=True)

seed(5489)

lamb_set = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
perfs_df = DataFrame()

for fold_idx in range(1, 11):

    for lamb in lamb_set :

        print(f'Training and validating with fold_idx = {fold_idx} and lamb = {lamb}')

        train_fpath = jnt(feats_base_dir, 'train' + str(fold_idx), 'data.mat')
        train_contents = loadmat(train_fpath)
        Ytrain = train_contents['features']
        Ttrain = train_contents['labels']

        valid_fpath = jnt(feats_base_dir, 'valid' + str(fold_idx), 'data.mat')
        valid_contents = loadmat(valid_fpath)
        Yvalid = valid_contents['features']
        Tvalid = valid_contents['labels'] 
        
        clf = DDF(lamb, 1)
        clf.fit(Ytrain.T, Ttrain.ravel())
        
        preds = clf.predict_proba(Yvalid.T).T
        perfs_df.loc[fold_idx, lamb] = roc_auc_score(Tvalid, preds[:,1])

print('Saving plot...')

xlabel = '$\lambda$'

xticklabels = ['{:.2f}'.format(lamb) for lamb in perfs_df.columns.to_list()]

plotKtuning(perfs_df, imgname='tuning_SRC.pdf', linewidth_frac=0.45, xlabel=xlabel, xticklabels=xticklabels, xticklabels_size=7.5, yticklabels_size=7.5)

print('Done!')
