import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from plot_utils import set_style

# df - dataframe
# title - figure title
def mylineplot( df, title):
    plt.title(title)
    ax = sns.lineplot(x="numframes", y="vals", hue='cols', data=df)
    plt.xlabel('Number of aggregated frames')
    plt.ylabel('EER')
    plt.ylim(0, 0.5)
    plt.xticks(np.arange(1, 6, step=1))
    plt.show()



# 
# Run from project root directory: python plots/lineplots.py
# 

eer_raw_s1=[0.41, 0.4, 0.4, 0.4, 0.4]
eer_raw_s2=[0.38, 0.38, 0.38, 0.37, 0.37]
eer_raw_cd=[0.46, 0.46, 0.46, 0.46, 0.45]

eer_ae_s1=[0.12, 0.08, 0.07, 0.06, 0.06]
eer_ae_s2=[0.09, 0.06, 0.05, 0.04, 0.04]
eer_ae_cd=[0.32, 0.25, 0.24, 0.23, 0.22]

eer_ee_s1=[0.08, 0.06, 0.05, 0.04, 0.04]
eer_ee_s2=[0.06, 0.04, 0.03, 0.03, 0.02]
eer_ee_cd=[0.27, 0.26, 0.25, 0.24, 0.24]

numframes = [ 1, 2, 3, 4, 5]

# SD - session 1
d_s1 = {'numframes': numframes, 'raw': eer_raw_s1, 'autoencoder':eer_ae_s1, 'end-to-end': eer_ee_s1}
df_s1  = pd.DataFrame(data=d_s1 )
df_s1 = df_s1.melt('numframes', var_name='cols',  value_name='vals')

# SD - session 2
d_s2 = {'numframes': numframes, 'raw': eer_raw_s2, 'autoencoder':eer_ae_s2, 'end-to-end': eer_ee_s2}
df_s2  = pd.DataFrame(data=d_s2 )
df_s2 = df_s2.melt('numframes', var_name='cols',  value_name='vals')

d_cd = {'numframes': numframes, 'raw': eer_raw_cd, 'autoencoder':eer_ae_cd, 'end-to-end': eer_ee_cd}
df_cd  = pd.DataFrame(data=d_cd )
df_cd = df_cd.melt('numframes', var_name='cols',  value_name='vals')


sns.set(rc={'figure.figsize':(4.5, 4.5)})
set_style()

mylineplot( df_s1, 'Same day - session 1')
mylineplot( df_s2, 'Same day - session 2')
mylineplot( df_cd, 'Cross day ')
    
