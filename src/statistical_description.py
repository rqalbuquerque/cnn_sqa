"""Statistics analysis of input csv.

"""
import math
import csv
import sys
import os.path

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import DataFrame, Series

def describe_two_columns(df, col1, col2, label):
  print('\n')
  print('**************************************** Describe ****************************************')
  print('Label: ' + label)
  print('**************************************** Describe ****************************************')
  print(df[[col1, col2]].describe())
  print('**************************************** ******** ****************************************')
  print('corr -----------------> ' + str(df[col1].corr(df[col2])))
  print('cov ------------------> ' + str(df[col1].cov(df[col2])))
  print('ks -------------------> ' + str(stats.ks_2samp(df[col1].values, df[col2].values)))
  print('skew(GT) -------------> ' + str(stats.skew(df[col1].values)))
  print('skew(Prediction) -----> ' + str(stats.skew(df[col2].values)))
  print('kurtosis(GT) ---------> ' + str(stats.kurtosis(df[col1].values)))
  print('kurtosis(Prediction) -> ' + str(stats.kurtosis(df[col2].values)))
  print('**************************************** Describe ****************************************')
  print('\n')

def plot_two_overlap_distributions(df, col1, col2, xlabel, qty_bins):
  fig, axes = plt.subplots(2,2, sharex=True)
  
  minv = min(df[col1].min(), df[col2].min())
  maxv = max(df[col1].max(), df[col2].max())
  nbins = np.linspace(minv, maxv, qty_bins)

  ax_gt = sns.distplot(df[col1], ax=axes[0,0], bins=nbins)
  axes[0,0].set_xlabel(xlabel)
  axes[0,0].set_title(col1)
  
  ax_pdt = sns.distplot(df[col2], ax=axes[1,0], bins=nbins)
  axes[1,0].set_xlabel(xlabel)
  axes[1,0].set_title(col2)

  ax_pdt = sns.distplot(df[col1], kde=False, ax=axes[0,1], bins=nbins)
  ax_pdt = sns.distplot(df[col2], kde=False, ax=axes[0,1], bins=nbins)
  axes[0,1].set_xlabel(xlabel)
  axes[0,1].set_title(col1 + ' and ' + col2)

  ax_pdt = sns.distplot(df[col1], hist=False, ax=axes[1,1], bins=nbins)
  ax_pdt = sns.distplot(df[col2], hist=False, ax=axes[1,1], bins=nbins)
  axes[1,1].set_xlabel(xlabel)
  axes[1,1].set_title(col1 + ' and ' + col2)

  return fig, axes


def plot_overlap_distributions(df, col1, col2, filters, filter_col, xlabel, bins, qty_rows, qty_cols):
  fig, axes = plt.subplots(qty_rows, qty_cols, sharex=True, sharey=True)
  
  for (i, elmo) in enumerate(filters):
    line, col = i/qty_cols,i%qty_cols
    minv = min(df[col1].min(), df[col2].min())
    maxv = max(df[col1].max(), df[col2].max())
    nbins = np.linspace(minv, maxv, bins)
    ax_pdt = sns.distplot(df[df[filter_col] == elmo][col1], ax=axes[line, col], bins=nbins)
    ax_pdt = sns.distplot(df[df[filter_col] == elmo][col2], ax=axes[line, col], bins=nbins)
    axes[line, col].set_xlabel(xlabel)
    axes[line, col].set_title(elmo)

  return fig, axes

if __name__ == '__main__':
    # test input args
    if len(sys.argv) < 2:
        raise ValueError('Invalid number of args!')
    elif not os.path.isfile(sys.argv[1]): 
        raise ValueError('Path is not a valid file!')

    file_name = sys.argv[1]
    parent_path = os.path.dirname(file_name)

    # read        
    df = pd.read_csv(file_name, delimiter=',')

    # global configs
    plt.rc('figure', figsize=(10, 8))
    sns.set(style="whitegrid")

    # Error
    # df['Error'] = abs(df['GT']-df['Prediction'])

    # df_gt = df[['Noise','SNR','GT']]
    # df_gt.loc[:,'Type'] = 'GroundTruth'
    # df_gt = df_gt.rename(index=str, columns={"GT": "Score"})

    # df_pdt = df[['Noise','SNR','Prediction']]
    # df_pdt.loc[:,'Type'] = 'Prediction'
    # df_pdt = df_pdt.rename(index=str, columns={"Prediction": "Score"})

    # df_by_type = pd.concat([df_gt,df_pdt])

    # df_samples = df.Name.unique()
    # df_noises = df.Noise.unique()
    # df_snrs = np.sort(df.SNR.unique())

    # score_ticks = [0,1,2,3,4,5]

    """ ************************************************ General ************************************************ """
    # describe_two_columns(df, 'GT', 'Prediction', 'General')

    # pairwise bivariate distributions
    # pg = sns.pairplot(df, vars=['GT', 'Prediction'], kind="reg")

    # sns.jointplot(x="GT", y="Prediction", data=df, kind="reg")
    # with sns.axes_style("white"):    
    #   sns.jointplot(x="GT", y="Prediction", data=df, kind="hex")
    # sns.jointplot(x="GT", y="Prediction", data=df, kind="kde")

    # rel = sns.relplot(x="GT", y="Prediction", hue='SNR', data=df, legend="full")
    # rel.set(xticks=score_ticks, yticks=score_ticks, title='GT x Predition (Scores)')

    # fig, axes = plot_two_overlap_distributions(df, 'GT', 'Prediction', 'Scores', 30)
    """ ************************************************ General ************************************************ """


    """ ************************************************ By Noise ************************************************ """
    # Describe
    # for noise in df_noises:
    #   describe_two_columns(df[df.Noise == noise], 'GT', 'Prediction', noise)

    # Distributions
    # pg = sns.pairplot(df[df['SNR'] == 5], vars=['GT', 'Prediction'], hue='Noise')

    # for noise in df_noises:
      # plot_overlap_distributions(df[df['Noise'] == noise], 'GT', 'Prediction', df_snrs, 'SNR', 'Scores', 30, 2, 3)

    # rel = sns.relplot(x="GT", y="Prediction", col="Noise", col_wrap=3, hue="SNR", data=df, legend="full")
    # rel.set(xticks=score_ticks, yticks=score_ticks)

    # Average increase of score by snr
    # rel = sns.relplot(x="SNR", y="Score", col="Noise", col_wrap=3, kind='line', hue="Type", style="Type", data=df_by_type, legend="full")
    # rel = sns.relplot(x="SNR", y="Prediction", kind='line', hue="Noise", style="Noise", legend="full", data=df)
    # rel = sns.relplot(x="SNR", y="GT", kind='line', hue="Noise", style="Noise", legend="full", data=df)

    # box plot
    # sns.catplot(x="SNR", y="Score", hue="Type", col="Noise", col_wrap=3, data=df_by_type)
    # sns.catplot(x="SNR", y="Score", hue="Type", col="Noise", col_wrap=3, kind="swarm", data=df_by_type)
    # sns.catplot(x="SNR", y="Score", hue="Type", col="Noise", col_wrap=3, kind="box", data=df_by_type)
    # sns.catplot(x="SNR", y="Score", hue="Type", col="Noise", col_wrap=3, kind="boxen", data=df_by_type)
    # fg = sns.catplot(x="SNR", y="Score", hue="Type", col="Noise", col_wrap=3, kind="violin", split=True, data=df_by_type)
    # sns.catplot(x="SNR", y="Score", hue="Type", col="Noise", col_wrap=3, kind="bar", data=df_by_type)

    # histograms
    # g = sns.FacetGrid(df_by_type[df_by_type['Noise'] == 'STREET'], hue="Type", col="SNR", col_wrap=3, margin_titles=True)
    # g.map(plt.hist, "Score", bins=np.linspace(1, 5, 30), alpha=.5).add_legend()

    # g = sns.FacetGrid(df_by_type, col="Noise", row="SNR", hue="Type", margin_titles=True, height=2, aspect=1.1)
    # g.map(sns.distplot, "Score", hist=False)

    # g = sns.FacetGrid(df_by_type, col="Noise", col_wrap=3, hue="Type", margin_titles=True, height=2, aspect=1.1)
    # g.map(sns.distplot, "Score", hist=False).add_legend()

    # g = sns.FacetGrid(df, col="Noise", col_wrap=3)
    # g.map(sns.boxplot, "SNR", "Error")

    # g = sns.PairGrid(df, vars=["GT", "Prediction"], hue="SNR")
    # g.map_diag(sns.kdeplot)
    # g.map_upper(plt.scatter)
    # g.map_lower(sns.kdeplot)
    # g.add_legend()

    """ ************************************************ By Noise ************************************************ """


    """ ************************************************ By SNR ************************************************ """
    # Describe
    # for snr in df_snrs:
    #   describe_two_columns(df[df.SNR == snr], 'GT', 'Prediction', snr)

    # Distributions
    # pg = sns.pairplot(df, vars=['GT', 'Prediction'], hue='SNR')
    # pg.set(xticks=score_ticks, yticks=score_ticks)

    # plot_overlap_distributions(df[df['SNR'] == 30], 'GT', 'Prediction', df_noises, 'Noise', 'Scores', 30, 2, 3)

    # box plot
    # sns.catplot(x="Noise", y="Score", hue="Type", col="SNR", col_wrap=3, data=df_by_type, kind="box")

    # rel = sns.relplot(x="Noise", y="Score", col="SNR", col_wrap=3, hue="Type", data=df_by_type, kind="violin")

    # histograms
    # g = sns.FacetGrid(df_by_type, row="Type", col="SNR", margin_titles=True)
    # g.map(plt.hist, "Score", color="steelblue", bins=np.linspace(1, 5, 40))
    """ ************************************************ By SNR ************************************************ """

    """ ************************************************ SNR x NOISE ************************************************ """
    # sns.catplot(x="SNR", y="Score", hue="Type", row="SNR", col="Noise", data=df_by_type, kind="strip")
    """ ************************************************ SNR x NOISE ************************************************ """

    """ ************************************************ By Sample ************************************************ """
    """ ************************************************ By Sample ************************************************ """

    plt.show()
    # plt.savefig(parent_path + '/test.png', dpi=400, bbox_inches='tight')
