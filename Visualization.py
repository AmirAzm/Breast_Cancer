#######################|PRIMARY SETUP|##################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from scipy import stats
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
########################################################################

########################|READ DATA|##################################
fe =[['Mean','Standard_error','Largest_value'],['Radius','Texture','Perimeter','Area','Smoothness','Compactness','Concavity','Concave_points','Symmetry','Fractal_dimension']]
fe = pd.MultiIndex.from_product(fe)
data = pd.read_csv('wdbc_train.data',names=['ID','Label',*fe])
########################| DELETE OUTLIERS|###########################
# z_score = np.abs(stats.zscore(data[fe]))
# data = data[(z_score < 3).all(axis=1)]
#####################################################################
Mean_Data = data[fe[:10]]
STD_Data = data[fe[10:20]]
Max_Data = data[fe[20:30]]
######################|DRAW PLOT|######################################
fig = plt.figure(figsize=(15,15))
for i in range(10):
    fig.add_subplot(5,2,i+1)
    sns.distplot(Max_Data.iloc[:,i],hist=True,kde=True,bins=int(180/5),color='darkblue',hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4})
    #plt.boxplot(Max_Data.iloc[:,i],vert=False)
    plt.title(Max_Data.iloc[:,i].name[1],loc='right')
    plt.xlabel(None)
    path = 'TESTRe/Train_'+Max_Data.iloc[:,0].name[0]+'_Boxplot(whitout Outliers).png'
    plt.savefig(path,dpi=300)
plt.show()
#######################################################################