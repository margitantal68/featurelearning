import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from util.utils import create_userids, print_list
from util.normalization import normalize_rows
import util.settings as st

import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

def generate_users_list(start_user, stop_user):
    users_list = []
    for i in range (start_user, stop_user):
        if( i<10 ):
            users_list.append('u00'+ str(i))
        if (i>=10 and i<100):
            users_list.append('u0'+str(i))
        if( i > 100 ):
            users_list.append('u'+str(i))
    return users_list

# converts a string to a number
# eg. u001 --> 1
def myfunc( str ):
    return (int)(str[1:])


def plot_tsne(input_name, output_fig_name, NUM_USERS, plot_title):
    df = pd.read_csv(input_name)
    rows, cols = df.shape


    df['user'] = df['user'].apply(lambda x: myfunc(x) )
    print(df.head)
    select_classes = [ i for i in range(1, NUM_USERS+1)]
    print(select_classes)
    df = df.loc[df[df.columns[-1]].isin(select_classes)]

    X = df.values
    X = X[:, 0:cols]
    y = X[:, -1]
    print(X.shape)
    print(y.shape)
    print(y)
    y = LabelEncoder().fit_transform(y)
    print(y)
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    X_2d = tsne.fit_transform(X)

    target_ids = np.unique(y)
    print(target_ids)
    fig = plt.figure()
   

    # colors = clrs = sns.color_palette('husl', n_colors=NUM_USERS) 
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'tan', 'orange', 'purple'
    for i, c, label  in zip(target_ids,colors, target_ids):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c = c, label=label)
    plt.legend()
    # for i in target_ids:
    #     plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1])
  
  
    # legendstr =[]
    # for i in range(1, NUM_USERS+1):
    #     legendstr.append("USER "+str(i))
    # plt.legend( legendstr)
    
    plt.title(plot_title)
    plt.savefig( output_fig_name)
    plt.show()

# generate a list of colors
def generate_colors(NUM_COLORS):
    cm = plt.get_cmap('gist_rainbow')


plot_tsne('output_csv/zju_session0_cycles_raw.csv', 'tsne_cycles_raw.png', 10, 'Raw features (cycles)')
# plot_tsne('output_csv/zju_session0_raw.csv', 'tsne_raw.png', 10, 'Raw features (frames)')
# plot_tsne('output_csv/session0_autoencoder.csv', 'tsne_autoencoder.png', 10, 'Autoencoder features')
# plot_tsne('output_csv/session0_endtoend.csv', 'tsne_endtoend.png', 10, 'End-to-end features')

# plot_tsne('output_csv/session0_autoencoder_rnd.csv', 'tsne_autoencoder_rnd.png', 10, 'Autoencoder (augm: rnd) features')
# plot_tsne('output_csv/session0_endtoend_rnd.csv', 'tsne_endtoend_rnd.png', 10, 'End-to-end (augm: rnd) features')

# plot_tsne('output_csv/session0_autoencoder_cshift.csv', 'tsne_autoencoder_cshift.png', 10, 'Autoencoder (augm: cshift) features')
# plot_tsne('output_csv/session0_endtoend_cshift.csv', 'tsne_endtoend_cshift.png', 10, 'End-to-end (augm: cshift) features')