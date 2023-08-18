
from matplotlib import rc, rcParams
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig, subplots
from numpy import arange, ceil, floor
from pandas import DataFrame, melt
import seaborn as sns

def plotKtuning(perfs_df, imgname='nome_da_imagem.png', linewidth_frac=1, xlabel='nome_do_hiperparametro', xticklabels='0', axes_labelsize=10, font_size=8, legend_size=8, xticklabels_size=8, yticklabels_size=8):

    # Essa função produz um gráfico referente ao ajuste de um hiperparâmetro de um determinado classificador com base nos ranqueamentos obtidos em K conjuntos de validação (por exemplo, figuras da seção 5.2.2 da tese). 

    # perfs_df: dataframe em que perfs_df.iloc[i,j] é o desempenho do classificador no i-ésimo conjunto de validação usando o j-ésimo hiperparâmetro avaliado. Obs.: assume-se que quanto maior o desempenho, melhor (se esse não for o caso, entrar com 1 - perfs_df). 
    
    # linewidth_frac é a fração do \linewidth que será usada no Latex. 

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
    
    
    
    #sns.set(rc=myrc)
    sns.set_style('whitegrid')
    
    #rc('text', usetex=True)
    #rc('font', **{'family': 'serif', 'serif':['Palatino']})

    xlocs = perfs_df.columns
    ylocs = arange(1, len(perfs_df.columns) + 1)

    ranks = perfs_df.rank(axis=1, ascending=False)
    pos_opt = ranks.mean(axis=0).argmin()

    xticklabels[pos_opt] = '\\textbf{{{:s}}}'.format(xticklabels[pos_opt])
    xticklabels = ['{:s}'.format(x.replace('.',',')) for x in xticklabels]
    
    yticklabels = ['{:.0f}$^{{\\underline{{o}}}}$'.format(y) for y in arange(1, len(perfs_df.columns) + 1)]

    df = melt(frame=ranks, var_name=xlabel, value_name='rank')

    freqs = DataFrame(index=ylocs, columns=xlocs)
    for col in ranks.columns:
        floored = ranks.loc[:,col].apply(lambda x: floor(x)).value_counts().reindex(ylocs, fill_value=0)
        ceiled = ranks.loc[:,col].apply(lambda x: ceil(x)).value_counts().reindex(ylocs, fill_value=0)
        freqs.loc[:,col] = (floored + ceiled)/2

    freqs /= perfs_df.shape[0] 
    
    for index, row in df.iterrows():
        df.loc[index,'size'] = freqs[row[xlabel]].loc[floor(row['rank']):ceil(row['rank'])].sum()

    for index, row in df.iterrows():
        df.loc[index,'x'] = perfs_df.columns.to_list().index(row[xlabel]) + 1

    plt.rcParams.update(tex_fonts)

    fig, ax = subplots(1, 1, figsize=set_size(width='thesis', fraction=linewidth_frac), dpi=600)
    
    sns.scatterplot(data=df, x='x', y='rank', size='size', hue='size', hue_norm=(0, 1), sizes=(10, 50), linewidth=0, palette=sns.color_palette('viridis', as_cmap=True))

    g = sns.lineplot(ax=ax, data=df, x='x', y='rank', color='orangered', ci='sd', marker=None, linewidth = 5, alpha=0.7, clip_on=False)

    g.legend([],[], frameon=False)

    g.set_xlabel(xlabel)
    g.set_ylabel('Colocação')

    g.set(xlim=(df['x'].min()-0.5,df['x'].max()+0.5))
    g.set(ylim=(df['rank'].min()-0.5,df['rank'].max()+0.5))

    ax.set_xticks(arange(1, len(perfs_df.columns) + 1), labels=xticklabels)
    ax.set_yticks(arange(1, len(perfs_df.columns) + 1), labels=yticklabels)

    g.invert_yaxis()
    
    #rcParams.update(myrc)

    savefig(imgname, transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)

def set_size(width, fraction=1, subplots=(1, 1)):

    # Essa é uma função auxiliar que determina automaticamente as dimensões da imagem no Matplotlib/Seaborn para que a mesma seja compatível com o Latex. Tal função foi baixada em: https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
