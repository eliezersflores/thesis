import sys

from numpy import log2
from numpy.random import seed
from pandas import DataFrame
from scipy.io import loadmat
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(1, '../../../aux')
from clfsettings import train_datasets
from pathutils import jnt
from plotutils import plotKtuning

train_name = '_'.join(train_datasets)
feats_base_dir = jnt('../..', 'feats', train_name, 'resnet50', checkparts=True)

seed(5489)

mtry_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
perfs_df = DataFrame()

for fold_idx in range(1, 11):

    for mtry in mtry_set:

        print(f'Training and validating with fold_idx = {fold_idx} and mtry = {mtry}')

        train_fpath = jnt(feats_base_dir, 'train' + str(fold_idx), 'data.mat')
        train_contents = loadmat(train_fpath)
        Ytrain = train_contents['features']
        Ttrain = train_contents['labels']

        valid_fpath = jnt(feats_base_dir, 'valid' + str(fold_idx), 'data.mat')
        valid_contents = loadmat(valid_fpath)
        Yvalid = valid_contents['features']
        Tvalid = valid_contents['labels'] 
 
        clf = RandomForestClassifier(n_estimators=200, max_features = mtry, n_jobs=-1)  
        clf.fit(Ytrain, Ttrain.ravel())
        
        preds = clf.predict_proba(Yvalid)
        perfs_df.loc[fold_idx, mtry] = roc_auc_score(Tvalid, preds[:,1])

print('Saving plot...')

xlabel = '$\\eta$'
xticklabels = ['{:.1f}'.format(eta) for eta in perfs_df.columns.to_list()]

df = plotKtuning(perfs_df, imgname='tuning_RF.pdf', linewidth_frac=0.45, xlabel=xlabel, xticklabels=xticklabels, xticklabels_size=7.5, yticklabels_size=7.5)

print('Done!')
