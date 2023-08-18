import sys

from numpy.random import seed
from pandas import DataFrame
from scipy.io import loadmat
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(1, '../../../aux')
from clfsettings import train_datasets
from pathutils import jnt
from plotutils import plotKtuning

train_name = '_'.join(train_datasets)
feats_base_dir = jnt('../..', 'feats', train_name, 'resnet50', checkparts=True)

seed(5489)

k_set = [1, 3, 5, 7, 9, 11, 13, 15]
perfs_df = DataFrame()

for fold_idx in range(1, 11):

    for k in k_set :

        print(f'Training and validating with fold_idx = {fold_idx} and k = {k}')

        train_fpath = jnt(feats_base_dir, 'train' + str(fold_idx), 'data.mat')
        train_contents = loadmat(train_fpath)
        Ytrain = train_contents['features']
        Ttrain = train_contents['labels']

        valid_fpath = jnt(feats_base_dir, 'valid' + str(fold_idx), 'data.mat')
        valid_contents = loadmat(valid_fpath)
        Yvalid = valid_contents['features']
        Tvalid = valid_contents['labels'] 
 
        clf = KNeighborsClassifier(n_neighbors=k, algorithm='brute')  
        clf.fit(Ytrain, Ttrain.ravel())
        
        preds = clf.predict_proba(Yvalid)
        perfs_df.loc[fold_idx, k] = roc_auc_score(Tvalid, preds[:,1])

print('Saving plot...')

xlabel = '$K$'
xticklabels = ['{:.0f}'.format(k) for k in perfs_df.columns.to_list()]

plotKtuning(perfs_df, imgname='tuning_KNN.pdf', linewidth_frac=0.45, xlabel=xlabel, xticklabels=xticklabels, xticklabels_size=7.5, yticklabels_size=7.5)

print('Done!')
