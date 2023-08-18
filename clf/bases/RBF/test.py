import sys

from numpy.random import seed
from pandas import read_csv
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

sys.path.insert(1, '../../../aux')
from clfsettings import train_datasets
from pathutils import jnt, lst, mkd

test_name = 'DermNet'
#test_name = 'MClass-ND'

train_name = '_'.join(train_datasets)
feats_base_dir = jnt('../..', 'feats', train_name, 'resnet50', checkparts=True)
folds_base_dir = jnt('../..', 'folds', train_name, checkparts=True)

seed(5489)

C = 2**-5
gamma = 2**-15

train_fpath = jnt(feats_base_dir, train_name, 'data.mat')
train_contents = loadmat(train_fpath)
Ytrain = train_contents['features']
Ttrain = train_contents['labels']

test_fpath = jnt(feats_base_dir, test_name, 'data.mat')
test_contents = loadmat(test_fpath)
Ytest = test_contents['features']
Ttest = test_contents['labels'] 

clf = SVC(kernel='linear', C=C, gamma=gamma, probability=True)  
clf.fit(Ytrain, Ttrain.ravel())

df = read_csv(jnt(folds_base_dir, test_name + '.csv'), header=None)
df.columns = ['dataset', 'label', 'img']
df['melscore'] = clf.predict_proba(Ytest)[:,1]
df.to_csv(jnt('..', 'Results', test_name + '_RBF.csv'), index=False)



