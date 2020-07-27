import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from util.utils import create_userids, print_list
from util.normalization import normalize_rows
import util.settings as st

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

def plot_ROC(userid, fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - user '+ userid)
    plt.legend(loc="lower right")
    plt.show()
    
def plot_histogram( values, number_of_bins = 10 ):
    plt.hist(values, bins = number_of_bins)
    plt.xlabel('Bins')
    plt.ylabel('Occurrence')
    plt.title('AUC distribution ')
    plt.show()

# precodition: OUTPUT_FIGURES folder must exists

def plot_raw_data( df, NUM_SAMPLES_PER_CLASS):
    userids = create_userids( df )
    NUM_USERS = len(userids)
    for i in range(0,NUM_USERS):
        userid = userids[i]
        print(userid)
        user_data = df.loc[df.iloc[:, -1].isin([userid])]
        # Select data for training
        user_data = user_data.drop(user_data.columns[-1], axis=1)
        user_array = user_data.values[0:NUM_SAMPLES_PER_CLASS,:]
        rows, cols = user_array.shape
        plt.clf()
        plt.xlabel('Time')
        plt.title("User "+str(userids[ i ])) 
        for row in range(rows):
            plt.plot(user_array[row,:])
        output_file = str(userids[ i ]) + '.png'
        print(output_file)
        plt.savefig(st.OUTPUT_FIGURES+"/"+output_file)
        
def plot_data_distribution( filename, plotname ):
    
    fig = plt.figure(figsize=(15, 12))

    # df = pd.read_csv(filename, header = None)
    # array = df.values
    # rows, cols = array.shape
    # L = np.ravel( array[:,0:cols-1] )
     
    # plt.hist(L, color='#3F5D7D')
    
    train = pd.read_csv(filename, header = None)
    cols = 5
    print("Average, Stdev, Min, Max")
    # loop over cols^2 vars
    for i in range(0, cols * cols):
        plt.subplot(cols, cols, i+1)
        f = plt.gca()
        # f.axes.get_yaxis().set_visible(False)
        # f.axes.set_ylim([0, train.shape[0]])

        # vals = np.size(train.iloc[:, i].unique())
        # if vals < 10:
        #     bins = vals
        # else:
        #     vals = 10

        # plt.hist(train.iloc[:, i], bins=30, color='#3F5D7D')
        mean_value = round(np.mean(train.iloc[:, i]), 2) 
        std_value = round(np.std(train.iloc[:, i]), 2)
        min_value = round(np.min(train.iloc[:, i]), 2)
        max_value = round(np.max(train.iloc[:, i]), 2)

        print(str(mean_value)+", "+ str(std_value) +", "+ str(min_value) +", "+ str(max_value) )
        plt.boxplot(train.iloc[:, i])

    plt.tight_layout()

    plt.savefig(plotname)
    plt.show()




# Plots t-SNE projections of the genuine and forged signatures
# input_csv/forgery_mcyt_1.csv
# input_csv/genuine_mcyt_1.csv
def plot_tsne_binary():
    df_genuine = pd.read_csv("input_csv/genuine_mcyt_1.csv", header =None)
    df_forgery = pd.read_csv("input_csv/forgery_mcyt_1.csv", header =None)
    
    NUM_USERS = 100
    for user in range(0, NUM_USERS):
        userlist = [ user ]
        df_user_genuine = df_genuine.loc[df_genuine[df_genuine.columns[-1]].isin(userlist)]
        df_user_forgery = df_forgery.loc[df_forgery[df_forgery.columns[-1]].isin(userlist)]

        print(df_user_genuine.shape)
        print(df_user_forgery.shape)

        G = df_user_genuine.values
        F = df_user_forgery.values

        df1 = pd.DataFrame(G[:,0:1024])
        df2 = pd.DataFrame(F[:,0:1024])

        df = pd.concat( [df1, df2] )
        

        print(df.shape)

        y = [0] * 25 + [1] * 25  
        y = LabelEncoder().fit_transform(y)
        X = df.values
        tsne = TSNE(n_components=2, init='random', random_state=41)
        X_2d = tsne.fit_transform(X, y)

    
    
    
        # target_ids = np.unique(y)
        for i in [0,1]:
            plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1])
        plt.legend(['Genuine', 'Forgery'])
    
        plt.title("User "+str(user))
        plt.show()


# Plots t-SNE projections of NUM_USERS signatures
# Shows the separability of the users based on raw data
# or other features

def plot_tsne(input_name, output_name, NUM_USERS, plot_title):
   
    df = pd.read_csv(input_name, header =None)
    rows, cols = df.shape

    select_classes = [ i for i in range(0, NUM_USERS)]
    print(select_classes)
    df = df.loc[df[df.columns[-1]].isin(select_classes)]

    X = df.values
    X = X[:, 0:cols]
    y = X[:, -1]
    print(y.shape)
    print(y)
    y = LabelEncoder().fit_transform(y)
    print(y)
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    X_2d = tsne.fit_transform(X)

    target_ids = np.unique(y)
    for i in target_ids:
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1])
    legendstr =[]
    for i in range(0, NUM_USERS):
        legendstr.append("USER "+str(i))
    plt.legend( legendstr)
    
    plt.title(plot_title)
    plt.savefig( output_name)
    plt.show()

    
    
