import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from plot_utils import set_style



# 
# Run from project root directory: python plots/histoplots.py
# 

df = pd.read_csv("input_csv/zju_gait_cycles2.csv")
# Column name Length
x = df['Length']

sns.set(rc={'figure.figsize':(5, 5)})
set_style()
sns.distplot(x, kde = False)
plt.title('ZJU-GaitAcc: histogram of gait cycles')
plt.xlabel('Cycle length [samples]')
plt.ylabel('Frequency')
plt.show()
