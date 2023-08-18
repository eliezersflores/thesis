from pickle import load
import sys

from datetime import datetime as dt
from matplotlib.pyplot import rc, rcParams
from numpy import abs, argmin, array, mean, sum
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from Orange.evaluation import compute_CD, graph_ranks

sys.path.insert(1, '../../aux')
from clfsettings import datasets, train_datasets
from pathutils import jnt, lst, mkd

alpha = 0.05

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rcParams.update({'font.size': 10})

train_name = '_'.join(train_datasets)

eval_results_dir = jnt('..', 'eval', train_name, checkparts=True)

with open(jnt(eval_results_dir, 'perfs.pkl'), 'rb') as f:
    perfs = load(f)

methods = perfs.pop('methods')
for key in perfs.keys():
    print(key)
    pfs = perfs[key]
    if key == 'MClass-ND':
        tot_hits = sum(pfs, axis=0)
        tot_hits -= tot_hits[5:].mean()
        tot_hits = abs(tot_hits)
        sel = argmin(tot_hits[5:])+5
        pfs[:,5] = pfs[:,sel]
        npfs = pfs[:,:6]
        npfs[:,5] = pfs[:,sel]
        methods2 = methods[:5]
        methods2.append('derm')
        pfs = npfs
    elif key == 'DermNet':
        methods2 = methods[:5]
    else:
        methods2 = methods[:5]
        for i, methodi in enumerate(methods2):
            for j, methodj in enumerate(methods2):
                if i < j:
                    print('{} x {}'.format(methodi, methodj))
                    print(wilcoxon(pfs[:,i], pfs[:,j], zero_method='zsplit'))
        continue
       
    print(friedmanchisquare(*pfs))
    print()

    ranks = array([rankdata(-p) for p in pfs])
    avg_ranks = mean(ranks, axis=0)

    cd = compute_CD(avg_ranks, pfs.shape[0], alpha=str(alpha), test='bonferroni-dunn')
    print('Critical difference = {:.2f}'.format(cd))

    graph_ranks(avg_ranks, methods2, cd=cd, width=len(methods2), textspace=1.6, cdmethod=0, filename='{:s}.png'.format(key))

