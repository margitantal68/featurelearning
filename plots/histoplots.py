import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

df = pd.read_csv("input_csv/zju_gait_cycles2.csv")
# Column name Length
x = df['Length']
sns.distplot(x, kde = False)
plt.title('ZJU-GaitAcc: histogram of gait cycles')
plt.xlabel('Cycle length [samples]')
plt.ylabel('Frequency')
plt.show()
