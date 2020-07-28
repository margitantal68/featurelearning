import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from plot_utils import set_style

# create a boxplot from a CSV
# 
# def csv2boxplot(filename, title, ylabel):
#     df = pd.read_csv( filename )
#     print(df.head)
#     # columns = ['raw','unsupervised', 'supervised']
#     columns = list(df.columns)
#     # df[columns].plot.box()
#     res = df.boxplot(column=columns, return_type='axes')
#     plt.title(title)
#     plt.ylabel(ylabel)
#     plt.show(res)


# create a boxplot from a dataframe
# 
def csv2boxplot(df, columns, title, ylabel):
    # columns = list(df.columns)
    # df[columns].plot.box()
    res = df.boxplot(column=columns, return_type='axes')
    plt.title(title)
    plt.xlabel('Type of features')
    plt.ylabel(ylabel)
    plt.show(res)


# create a boxplot from a dataframe
# 
def csv2boxplot_all(df, columns, title, ylabel):
    box = plt.boxplot(df, patch_artist=True)
    colors = ['blue', 'green', 'purple', 'blue', 'green', 'purple', 'blue', 'green', 'purple' ]
 
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.title(title)
    plt.xlabel('Type of features')
    plt.ylabel(ylabel)
    plt.show()
    

def snsboxplot(df, columns, title, ylabel):
    # sns.set(style="darkgrid")
    set_style()
    res = sns.boxplot(x="variable", y="value", data= pd.melt(df))
    plt.title(title)
    plt.xlabel('Type of features')
    plt.ylabel(ylabel)
    plt.show(res)



# 
# Run from project root directory: 
#                                   python plots/boxplots.py

df_raw =  pd.read_csv( 'results/raw_scores.csv' )
df_autoencoder =  pd.read_csv( 'results/autoencoder_scores.csv' )
df_endtoend    =  pd.read_csv( 'results/endtoend_scores.csv' )

columns =['raw', 'autoencoder', 'end-to-end']

sns.set(rc={'figure.figsize':(4.5, 4.5)})
set_style()

# session 1
data = [ df_raw['AUC1'], df_autoencoder['AUC1'], df_endtoend['AUC1']]
df = pd.concat(data, axis = 1, keys= columns)
snsboxplot(df, columns, 'Same-day evaluation - session 1', 'AUC' )

# session 2
data = [ df_raw['AUC2'], df_autoencoder['AUC2'], df_endtoend['AUC2']]
df = pd.concat(data, axis = 1, keys= columns)
snsboxplot(df, columns, 'Same-day evaluation - session 2', 'AUC' )


# cross-day
data = [ df_raw['AUC_cross'], df_autoencoder['AUC_cross'], df_endtoend['AUC_cross']]
df = pd.concat(data, axis = 1, keys= columns)
snsboxplot(df, columns, 'Cross-day evaluation', 'AUC' )


# ALL data
# data = [ df_raw['AUC1'], df_autoencoder['AUC1'], df_endtoend['AUC1'],  df_raw['AUC2'], df_autoencoder['AUC2'], df_endtoend['AUC2'], df_raw['AUC_cross'], df_autoencoder['AUC_cross'], df_endtoend['AUC_cross']]
# columns =['SD1-raw', 'SD1-autoenc', 'SD1-endtoend', 'SD2-raw', 'SD2-autoenc', 'SD2-endtoend', 'CD-raw', 'CD-autoenc', 'CD-endtoend']
# df = pd.concat(data,  axis = 1)

# csv2boxplot_all(df, columns, 'Performance comparison on ZJU-GaitAcc', 'AUC' )