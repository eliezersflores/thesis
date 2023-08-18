import sys

from matplotlib.pyplot import legend, plot, rc, rcParams, show, title, savefig, subplots, xlabel, xlim, ylabel, ylim
from numpy import arange, array, sqrt
from pandas import DataFrame, read_csv
from seaborn import color_palette, lineplot
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve, roc_auc_score

sys.path.insert(1, '../../../aux')
from clfsettings import datasets, train_datasets
from pathutils import jnt, lst, mkd
from plotutils import set_size
from strutils import lst2str

train_name = '_'.join(train_datasets)

test_results_dir = jnt('..', '..', 'tests', train_name, checkparts=True)
sota_results_dir = jnt('..', 'results')

# Essa função plota a curva ROC para algum método.
# Tal função é invocada várias vezes para gerar a figura da tese. 
# Quando der tempo, passar essa função (ou montar uma mais elaborada) para dentro do módulo plotutils. 
def plot_roc(df, label, color):
    fpr, tpr, th = roc_curve(df['label'], df['melscore'], pos_label='melanoma')
    roc_auc = auc(fpr, tpr)
    label = label + ' (AUC = {:.2f})'.format(roc_auc).replace('.', ',')
    g = lineplot(x=fpr, y=tpr, label=label, lw=3, color=color, alpha=0.9, ci=None)
    g.set_xlabel('TFP')
    g.set_ylabel('TVP')
    return fpr, tpr, th

def metric(fpr, tpr):
    spec = 1 - fpr
    sens = tpr
    gmean = sqrt(spec*sens)
    #return max(0.6*sens + 0.4*spec)
    #return max(sqrt(sqrt(sens *spec))) #gmean
    return gmean

# EDIT HERE
#dataset = 'DermNet'
dataset = 'MClass-ND'

imgname = 'roc_sota_' + dataset + '.pdf'
linewidth_frac=0.49

axes_labelsize=10
font_size=8
legend_size=5.5
xticklabels_size=8
yticklabels_size=8

tex_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "axes.labelsize": axes_labelsize,
    "font.size": font_size,
    "legend.fontsize": legend_size,
    "xtick.labelsize": xticklabels_size, 
    "ytick.labelsize": yticklabels_size 
}

sns.set_style('whitegrid')  

rcParams.update(tex_fonts)

fig, ax = subplots(1, 1, figsize=set_size(width='thesis', fraction=linewidth_frac), dpi=600)

df = read_csv(jnt(sota_results_dir, dataset + '_ARL.csv'))
fpr, tpr, th = plot_roc(df, 'ARL', color_palette()[4])
m1 = metric(fpr, tpr)
#print('gmean(ARL) = ', metric(fpr, tpr))

df = read_csv(jnt(sota_results_dir, dataset + '_DW.csv'))
fpr, tpr, th = plot_roc(df, 'DW', color_palette()[1])
m2 = metric(fpr, tpr)
#print('gmean(DW) = ', metric(fpr, tpr))

df = read_csv(jnt(sota_results_dir, dataset + '_IW.csv'))
fpr, tpr, th = plot_roc(df, 'IW', color_palette()[0])
m3 = metric(fpr, tpr)
#print('gmean(IW) = ', metric(fpr, tpr))

df = read_csv(jnt(sota_results_dir, dataset + '_SLA.csv'))
fpr, tpr, th = plot_roc(df, 'SLA', color_palette()[2])
m4 = metric(fpr, tpr)
#print('gmean(SLA) = ', metric(fpr, tpr))

df = read_csv(jnt(test_results_dir, dataset + '_resnet50_5_0.8.csv'))
fpr, tpr, th = plot_roc(df, 'método proposto', color_palette()[3])
m5 = metric(fpr, tpr)
#print('gmean(proposed) = ', metric(fpr, tpr))

xticklabels = lst2str(arange(0, 1.1, 0.2), q=1)
yticklabels = lst2str(arange(0, 1.1, 0.2), q=1)
xticklabels = ['{:s}'.format(x.replace('.',',')) for x in xticklabels]
yticklabels = ['{:s}'.format(y.replace('.',',')) for y in yticklabels]
ax.set_xticks(arange(0, 1.1, 0.2), labels=xticklabels)
ax.set_yticks(arange(0, 1.1, 0.2), labels=yticklabels)

savefig(imgname, transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)


