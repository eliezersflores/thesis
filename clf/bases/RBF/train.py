import sys

from numpy import log2 
from numpy.random import seed
from pandas import DataFrame
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

sys.path.insert(1, '../../../aux')
from clfsettings import train_datasets
from pathutils import jnt

train_name = '_'.join(train_datasets)
feats_base_dir = jnt('../..', 'feats', train_name, 'resnet50', checkparts=True)

seed(5489)

C_set = [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15] 
gamma_set = ([2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2**1, 2**3])
aucs = DataFrame()

for fold_idx in range(1, 11):

    train_fpath = jnt(feats_base_dir, 'train' + str(fold_idx), 'data.mat')
    train_contents = loadmat(train_fpath)
    Ytrain = train_contents['features']
    Ttrain = train_contents['labels']

    valid_fpath = jnt(feats_base_dir, 'valid' + str(fold_idx), 'data.mat')
    valid_contents = loadmat(valid_fpath)
    Yvalid = valid_contents['features']
    Tvalid = valid_contents['labels'] 

    for C in C_set:

        for gamma in gamma_set:
            
            print('Computing for fold_idx = {:d}, C = 2**{:.0f} and gamma = 2**{:.0f}'.format(fold_idx, round(log2(C)), round(log2(gamma))))

            clf = SVC(kernel='linear', C=C, gamma=gamma, probability=True)  
            clf.fit(Ytrain, Ttrain.ravel())
            
            preds = clf.predict_proba(Yvalid)
            aucs.loc[fold_idx, str(C) + '_' + str(gamma)] = roc_auc_score(Tvalid, preds[:,1])

print('\naucs:')
print(aucs.to_string())

ranks = aucs.rank(axis=1, ascending=False)
print('\nranks:')
print(ranks.to_string())

avranks = ranks.mean(axis=0)
print('\naverage ranks:')
print(avranks.to_string())

pars_opt = aucs.columns[avranks.argmin()]
C_opt = float(pars_opt.split('_')[0])
gamma_opt = float(pars_opt.split('_')[1])

print('\noptimal C = 2**{:.0f}'.format(round(log2(C_opt))))
print('\noptimal gamma = 2**{:.0f}'.format(round(log2(gamma_opt))))
