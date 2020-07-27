import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

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
    sns.set(style="darkgrid")
    res = sns.boxplot(x="variable", y="value", data= pd.melt(df))
    # df.boxplot(column=columns, return_type='axes')
    plt.title(title)
    plt.xlabel('Type of features')
    plt.ylabel(ylabel)
    plt.show(res)


def snsboxplot2(df, columns, title, ylabel):
    sns.set(style="darkgrid")
    df_long = pd.melt(df, "b", var_name = "variable", value_name = "value")
    res = sns.boxplot(x="variable", hue="b", y="value", data= df_long)
    plt.title(title)
    plt.xlabel('Type of features')
    plt.ylabel(ylabel)
    plt.show(res)


df_raw =  pd.read_csv( 'results/raw_scores.csv' )
df_autoencoder =  pd.read_csv( 'results/autoencoder_scores.csv' )
df_endtoend    =  pd.read_csv( 'results/endtoend_scores.csv' )

# columns = ['AUC1', 'EER1', 'AUC2', 'EER2', 'AUC_cross', 'EER_cross']
columns =['raw', 'autoencoder', 'end-to-end']

# ALL data
# data = [ df_raw['AUC1'], df_autoencoder['AUC1'], df_endtoend['AUC1'],  df_raw['AUC2'], df_autoencoder['AUC2'], df_endtoend['AUC2'], df_raw['AUC_cross'], df_autoencoder['AUC_cross'], df_endtoend['AUC_cross']]
# columns =['SD1-raw', 'SD1-autoenc', 'SD1-endtoend', 'SD2-raw', 'SD2-autoenc', 'SD2-endtoend', 'CD-raw', 'CD-autoenc', 'CD-endtoend']
# df = pd.concat(data,  axis = 1)

# csv2boxplot_all(df, columns, 'Performance comparison on ZJU-GaitAcc', 'AUC' )


# session 1
# data = [ df_raw['AUC1'], df_autoencoder['AUC1'], df_endtoend['AUC1']]
# df = pd.concat(data, axis = 1, keys= columns)
# # csv2boxplot(df, columns, 'Same-day evaluation - session 1', 'AUC' )
# snsboxplot(df, columns, 'Same-day evaluation - session 1', 'AUC' )

# session 2
# data = [ df_raw['AUC2'], df_autoencoder['AUC2'], df_endtoend['AUC2']]
# df = pd.concat(data, axis = 1, keys= columns)
# csv2boxplot(df, columns, 'Same-day evaluation - session 2', 'AUC' )
# snsboxplot(df, columns, 'Same-day evaluation - session 2', 'AUC' )

data = [df_raw['AUC1'], df_raw['AUC2'], df_autoencoder['AUC1'], df_autoencoder['AUC2'], df_endtoend['AUC1'], df_endtoend['AUC2']]
df = pd.concat(data, axis=1, keys=['a', 'b', 'c', 'd', 'e', 'f'])
snsboxplot2(df, ['a', 'b', 'c', 'd', 'e', 'f'], 'valami', 'valami')


# cross-day
# data = [ df_raw['AUC_cross'], df_autoencoder['AUC_cross'], df_endtoend['AUC_cross']]
# df = pd.concat(data, axis = 1, keys= columns)
# # csv2boxplot(df, columns, 'Cross-day evaluation', 'AUC' )
# snsboxplot(df, columns, 'Cross-day evaluation', 'AUC' )


