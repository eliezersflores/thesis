import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import wilcoxon, friedmanchisquare, rankdata
from Orange.evaluation import graph_ranks

sys.path.insert(1, '../../aux')
from clfsettings import train_datasets
from evalutils import compute_avranks, compute_CD, compute_wins 
from pathutils import jnt, lst, mkd
from strutils import df2latex, lst2str, method2prepstr

print_wins = False

train_name = '_'.join(train_datasets)

feats_base_dir = jnt('..', 'feats', train_name, checkparts=True)

filename = 'results_DermIS_DermQuest.csv'
foldname = 'DermIS_DermQuest__2022_Jan_10_21_21_09'
fpath = jnt('..', 'train', foldname, filename) 

alpha = 0.05

df = pd.read_csv(fpath, names=['algorithm', 'auc1', 'auc2', 'auc3', 'auc4', 'auc5', 'auc6', 'auc7', 'auc8', 'auc9', 'auc10', 'bacc1', 'bacc2', 'bacc3', 'bacc4', 'bacc5', 'bacc6', 'bacc7', 'bacc8', 'bacc9', 'bacc10', 'gmean1', 'gmean2', 'gmean3', 'gmean4', 'gmean5', 'gmean6', 'gmean7', 'gmean8', 'gmean9', 'gmean10'])

df = df.drop(columns=['bacc1', 'bacc2', 'bacc3', 'bacc4', 'bacc5', 'bacc6', 'bacc7', 'bacc8', 'bacc9', 'bacc10', 'gmean1', 'gmean2', 'gmean3', 'gmean4', 'gmean5', 'gmean6', 'gmean7', 'gmean8', 'gmean9', 'gmean10'], axis=1)

algorithms_names = df['algorithm']
df = df.drop('algorithm', axis=1)

performances_array = df.values.T

#avranks = compute_avranks(performances_array)
#opt_idx = avranks.argmin()

d = dict()
for name in algorithms_names:
    cnn = name.split('_')[0]
    d[cnn] = pd.DataFrame()

for name, avranks, wins in zip(algorithms_names, compute_avranks(performances_array), compute_wins(performances_array)):
    cnn, lamb, vphi = name.split('_')
    if wins != 0 and print_wins == True:
        s = '{:7.2f} ({:.0f})'.format(avranks, wins)
    else:
        s = '{:7.2f}'.format(avranks, wins)
    d[cnn].loc[lamb, vphi] = s

    
tops = []
subcap_c1 = ''
subcap_c2 = 'a'

with open('tuning_all.txt', 'w') as f:

    for cnn, df in d.items():

        # define the subcaption subsequent to (z) as (ab)
        if subcap_c2 > 'z':
            subcap_c1 = 'a'
            subcap_c2 = 'a'

        # select the optimal result in df 
        sel_row, sel_col = np.unravel_index(df.applymap(lambda s: float(s.strip().split(' ')[0])).to_numpy().argmin(), df.shape)
        indexes = df.index.to_numpy()
        sel_lamb = indexes[sel_row]
        sel_phi = df.columns[sel_col]
        tops.append(f'{cnn}_{sel_lamb}_{sel_phi}')

        # put the aforementioned result to bold
        indexes[sel_row] = '\\textbf{' + sel_lamb + '}'
        df.set_index(indexes)
        df = df.rename(columns={df.columns[sel_col]:'\\textbf{' + sel_phi + '}'})
        df.iloc[sel_row, sel_col] = '\\textbf{' + df.iloc[sel_row, sel_col] + '}'

        # set ',' instead of '.' as separator
        df.index = df.index.map(lambda lamb: lamb.replace('.', ','))
        df.rename(columns=lambda vphi: vphi.replace('.', ','), inplace=True)
        df = df.applymap(lambda s: s.replace('.', ','))

        # generate the main latex code corresponding to df
        s = 'C{0.1\\linewidth}' + df.shape[1]*'C{0.12\\linewidth}'
        text = df.to_latex(column_format=s, escape=False, longtable=True)

        # preprocess and write in the destination file 
        text = text.replace('{}', '\\diagbox{$\\lambda$}{$\\varphi$}')
        text = text.replace('Continued on next page', '(continua na próxima página)')
        f.writelines(text)
        f.write('\\vspace{-4ex}\n')
        f.write('\\begin{table}[H]\n')
        f.write('\\caption*{(' + subcap_c1 + subcap_c2 + ') ' + method2prepstr(cnn) + '.}\n')
        f.write('\\end{table}\n\n')
        f.write('\\addtocounter{table}{-1}')

        # go to the next ascii character
        subcap_c2 = chr(ord(subcap_c2) + 1)
 
#wins = compute_wins(performances_array)

sel_idxs = []
for algorithm in tops:
    sel_idxs.append(algorithms_names.tolist().index(algorithm))
pfs = performances_array[:,sel_idxs]

ranks = np.array([rankdata(-p) for p in pfs])
avg_ranks = np.mean(ranks, axis=0)

cnns = [method.split('_')[0] for method in tops]

inputs = [224, 224, 224, 224, 240, 260, 300, 380, 456, 528, 600, 299, 299, 224, 224, 224, 224, 331, 224, 224, 224, 224, 224, 224, 224, 224, 224, 299]

n_features = []
for cnn in cnns:
    mat = loadmat(jnt(feats_base_dir, cnn, train_name, 'data.mat'))
    n_features.append(mat['features'].shape[1])

df = pd.DataFrame()
df['DCNN'] = [method2prepstr(method.split('_')[0]) for method in tops]
df['entrada $\\rightarrow$ características'] = ['{:d} $\\times$ {:d} $\\times$ 3 $\\rightarrow$ {:d}'.format(i, i, n) for i, n in zip(inputs, n_features)]
df['$\\lambda$'] = [float(method.split('_')[1]) for method in tops]
df['$\\varphi$'] = [float(method.split('_')[2]) for method in tops]

if print_wins == True:
    df['desempenho'] = [f'{avg_rank} ({win})' for avg_rank, win in zip(lst2str(avg_ranks, 1), lst2str(compute_wins(pfs), 0))]
else:
    df['desempenho'] = [f'{avg_rank}' for avg_rank in lst2str(avg_ranks, 1)]

df2latex(df, 'tuning_tops.txt')


# Hypothesis tests:

print(friedmanchisquare(*pfs.T))
print()

cd = compute_CD(pfs.shape[1], pfs.shape[0], alpha, 'nemenyi')
#cd = compute_CD(pfs.shape[1], pfs.shape[0], alpha, 'bonferroni-dunn')

print('Critical difference = {:.2f}'.format(cd))

#graph_ranks(avg_ranks, tops, cd=cd, width=len(tops), textspace=1.6, cdmethod=avg_ranks.argmin(), filename='{:s}.png'.format('hypothesis_test'))

graph_ranks(avg_ranks, tops, cd=cd, width=len(tops), textspace=1.6, filename='ht_tops.png')
#plt.show()
