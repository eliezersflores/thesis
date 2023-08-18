import sys

from numpy.random import seed
from pandas import Series
from scipy.io import loadmat
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(1, '../../../aux')
from clfsettings import train_datasets
from pathutils import jnt, lst, mkd
from plotutils import plotKtuning

train_name = '_'.join(train_datasets)
feats_base_dir = jnt('../..', 'feats', train_name, 'resnet50', checkparts=True)
folds_base_dir = jnt('../..', 'folds', train_name, checkparts=True)

results_dir = jnt('..', 'Results'); mkd(results_dir)

seed(5489)

k_opt = 11 # optimized with the 'tuning.py' script. 

perfs = Series()

for fold_idx in range(1, 11):

    train_fpath = jnt(feats_base_dir, 'train' + str(fold_idx), 'data.mat')
    train_contents = loadmat(train_fpath)
    Ytrain = train_contents['features']
    Ttrain = train_contents['labels']

    valid_fpath = jnt(feats_base_dir, 'valid' + str(fold_idx), 'data.mat')
    valid_contents = loadmat(valid_fpath)
    Yvalid = valid_contents['features']
    Tvalid = valid_contents['labels'] 

    clf = KNeighborsClassifier(n_neighbors=k_opt, algorithm='brute')  
    clf.fit(Ytrain, Ttrain.ravel())
    
    preds = clf.predict_proba(Yvalid)
    perfs[str(fold_idx)] = roc_auc_score(Tvalid, preds[:,1])

perfs.to_csv(jnt(results_dir, 'KNN.csv'), index=False, header=False)
