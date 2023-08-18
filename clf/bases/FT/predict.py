import sys

from cv2 import INTER_NEAREST, resize
from numpy import expand_dims
from numpy.random import seed
from pandas import read_csv, Series
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
#from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import load_model

sys.path.insert(1, '../../../aux')
from clfsettings import labels, train_datasets
from imgutils import get_img
from pathutils import jnt, mkd

train_name = '_'.join(train_datasets)
folds_base_dir = jnt('..', '..', 'folds', train_name, checkparts=True)

results_dir = jnt('..', 'Results'); mkd(results_dir)

cnn = load_model(jnt('.', 'best_model.h5'))

perfs = Series()

for fold_idx in range(1, 11):

    print(f'\nWorking on fold {fold_idx} of 10...\n')

    df = read_csv(jnt(folds_base_dir, f'valid{fold_idx}.csv'), header=None, names=['dataset', 'label', 'img'])
    n_imgs = len(df)


    for i in range(n_imgs):
        print(f'Predicting the melanoma score for the image {i + 1} of {n_imgs}... ')
        dataset = df.loc[i, 'dataset']
        label = df.loc[i, 'label']
        image = df.loc[i, 'img']
        fpath = jnt('..', '..', '..', dataset, label, image + '.jpg')
        img = expand_dims(resize(get_img(fpath), (224,224), INTER_NEAREST)/255, axis=0)
        df.loc[i, 'melscore'] = cnn.predict(img)[:,0]

    preds = df['melscore']
    Tvalid = [int(x) for x in df['label'] == 'melanoma']
    perfs[str(fold_idx)] = roc_auc_score(Tvalid, preds)

perfs.to_csv(jnt(results_dir, 'FT.csv'), index=False, header=False)
