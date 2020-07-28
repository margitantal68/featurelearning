import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from plot_utils import set_style



# converts a string to a number
# eg. u001 --> 1
def myfunc( str ):
    return (int)(str[1:])


def plot_tsne(input_name, output_fig_name, NUM_USERS, plot_title):
    df = pd.read_csv(input_name)
    rows, cols = df.shape

    df['user'] = df['user'].apply(lambda x: myfunc(x) )
    select_classes = [ i for i in range(1, NUM_USERS+1)]
    df = df.loc[df[df.columns[-1]].isin(select_classes)]

    X = df.values
    X = X[:, 0:cols]
    y = X[:, -1]
    y = LabelEncoder().fit_transform(y)
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    X_2d = tsne.fit_transform(X)

    target_ids = np.unique(y)
    fig = plt.figure()
   
    # colors = clrs = sns.color_palette('husl', n_colors=NUM_USERS) 
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'tan', 'orange', 'purple'
    for i, c, label  in zip(target_ids,colors, target_ids):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c = c, label=label)
    # plt.legend()
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


# 
# Run from project root directory: 
#                                   python plots/tsneplots.py

sns.set(rc={'figure.figsize':(3, 3)})
set_style()

plot_tsne('output_csv/session0_raw.csv', 'tsne_raw.png', 10, 'Raw features')
plot_tsne('output_csv/session0_autoencoder.csv', 'tsne_autoencoder.png', 10, 'Autoencoder features')
plot_tsne('output_csv/session0_endtoend.csv', 'tsne_endtoend.png', 10, 'End-to-end features')

# plot_tsne('output_csv/session0_autoencoder_rnd.csv', 'tsne_autoencoder_rnd.png', 10, 'Autoencoder (augm: rnd) features')
# plot_tsne('output_csv/session0_endtoend_rnd.csv', 'tsne_endtoend_rnd.png', 10, 'End-to-end (augm: rnd) features')

# plot_tsne('output_csv/session0_autoencoder_cshift.csv', 'tsne_autoencoder_cshift.png', 10, 'Autoencoder (augm: cshift) features')
# plot_tsne('output_csv/session0_endtoend_cshift.csv', 'tsne_endtoend_cshift.png', 10, 'End-to-end (augm: cshift) features')