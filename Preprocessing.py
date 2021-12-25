#######################|PRIMARY SETUP|##################################
import pandas as pd
import numpy as np
from scipy import stats
import sys
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
#####################################################################

########################|READ DATA|##################################
fe =[['Mean','Standard_error','Largest_value'],['Radius','Texture','Perimeter','Area','Smoothness','Compactness','Concavity','Concave_points','Symmetry','Fractal_dimension']]
fe = pd.MultiIndex.from_product(fe)
data = pd.read_csv('wdbc_train.data',names=['ID','Label',*fe])
data_test = pd.read_csv('wdbc_test.data',names=['ID','Label',*fe])
print('Train Data Size :',data.shape)
print('Test Data Size :',data_test.shape)
#####################################################################

######################|TRAIN DATA PREPROCESSING|#######################
data_describe = data[fe].describe()
data_types = data.dtypes
data_miss = data.apply(lambda x: sum(x.isnull()),axis=0)
if not data_miss.any():
    print('There are no MISSING value in Train Data')
data_Mean_relation = data[fe[:10]].corr(method='pearson')
data_Std_relation = data[fe[10:20]].corr(method='pearson')
data_Max_relation = data[fe[20:30]].corr(method='pearson')
Q1 =data.quantile(0.25)
Q3 =data.quantile(0.75)
IQR = Q3 - Q1
data_outlier = (data < (Q1 - 1.5 * IQR))|(data > (Q3 + 1.5 * IQR))
z_score = np.abs(stats.zscore(data[fe]))
data_without_outlier = data[(z_score < 3).all(axis=1)]
######################################################################

######################|TEST DATA PREPROCESSING|#######################
data_T_describe = data_test[fe].describe()
data_T_types = data_test.dtypes
data_T_miss = data_test.apply(lambda x: sum(x.isnull()),axis=0)
if not data_T_miss.any():
    print('There are no MISSING value in Test Data')
data_T_Mean_relation = data_test[fe[:10]].corr(method='pearson')
data_T_Std_relation = data_test[fe[10:20]].corr(method='pearson')
data_T_Max_relation = data_test[fe[20:30]].corr(method='pearson')
Q1_T =data_test.quantile(0.25)
Q3_T =data_test.quantile(0.75)
IQR_T = Q3_T - Q1_T
data_T_outlier = (data_test < (Q1_T - 1.5 * IQR_T))|(data_test > (Q3_T + 1.5 * IQR_T))
z_score_T = np.abs(stats.zscore(data_test[fe]))
data_T_without_outlier = data_test[(z_score_T < 3).all(axis=1)]
######################################################################

########################|SAVE TRAIN DATA RESULTS|#####################
data.to_excel('TESTRe/Train_Preprocessing.xlsx',sheet_name='Data')
with pd.ExcelWriter('TESTRe/Train_Preprocessing.xlsx',mode='a') as wr:
    data_types.to_excel(wr, sheet_name='Types')
    data_describe.to_excel(wr,sheet_name='Describe')
    data_miss.to_excel(wr,sheet_name='Missing_value')
    data_Mean_relation.to_excel(wr,sheet_name='Mean_Correlation')
    data_Std_relation.to_excel(wr, sheet_name='STD_Correlation')
    data_Max_relation.to_excel(wr, sheet_name='MAX_Correlation')
    IQR.to_excel(wr,sheet_name='IQR')
    data_outlier.to_excel(wr,sheet_name='Outlires_IQR')
    data_without_outlier.to_excel(wr,'DataWithout_Outlier')
    pd.DataFrame(z_score,columns=fe).to_excel(wr,sheet_name='Z_Score')
######################################################################

########################|SAVE TEST DATA RESULTS|#####################
data_test.to_excel('TESTRe/Test_Preprocessing.xlsx',sheet_name='Data')
with pd.ExcelWriter('TESTRe/Test_Preprocessing.xlsx',mode='a') as wr:
    data_T_types.to_excel(wr, sheet_name='Types')
    data_T_describe.to_excel(wr,sheet_name='Describe')
    data_T_miss.to_excel(wr,sheet_name='Missing_value')
    data_T_Mean_relation.to_excel(wr,sheet_name='Mean_Correlation')
    data_T_Std_relation.to_excel(wr, sheet_name='STD_Correlation')
    data_T_Max_relation.to_excel(wr, sheet_name='MAX_Correlation')
    IQR_T.to_excel(wr,sheet_name='IQR')
    data_T_outlier.to_excel(wr,sheet_name='Outlires_IQR')
    data_T_without_outlier.to_excel(wr,'DataWithout_Outlier')
    pd.DataFrame(z_score_T,columns=fe).to_excel(wr,sheet_name='Z_Score')
######################################################################




