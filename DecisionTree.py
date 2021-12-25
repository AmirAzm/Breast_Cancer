#######################|PRIMARY SETUP|##################################
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,KFold,cross_validate
import sklearn.metrics as metrics
from scipy import stats
import sys

import graphviz
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
##########################################################################
#######################|READ DATA|########################################
fe =[['Mean','Standard_error','Largest_value'],['Radius','Texture','Perimeter','Area','Smoothness','Compactness','Concavity','Concave_points','Symmetry','Fractal_dimension']]
fe = pd.MultiIndex.from_product(fe)
features = ['Radius','Texture','Perimeter','Area','Smoothness','Compactness','Concavity','Concave_points','Symmetry','Fractal_dimension']
features_d = ['Radius','Texture','Smoothness','Compactness','Concavity','Concave_points','Symmetry','Fractal_dimension']
data = pd.read_csv('wdbc_train.data',names=['ID','Label',*fe])
data_test = pd.read_csv('wdbc_test.data',names=['ID','Label',*fe])
######################::|::DELETE-OUTLIERS::|::#######################################
# z_score = np.abs(stats.zscore(data[fe]))
# data = data[(z_score < 3).all(axis=1)]
# z_score_T = np.abs(stats.zscore(data_test[fe]))
# data_test = data_test[(z_score_T < 3).all(axis=1)]
#############################################################################
#######################::|::DIMENATION-REDUCTION::|::######################################
# del_columns = [['Mean','Standard_error','Largest_value'],['Perimeter','Area']]
# del_columns = pd.MultiIndex.from_product(del_columns)
# data = data.drop(columns=del_columns)
# data_test = data_test.drop(columns=del_columns)
# remain_col = [['Mean','Standard_error','Largest_value'],['Radius','Texture','Smoothness','Compactness','Concavity','Concave_points','Symmetry','Fractal_dimension']]
# remain_col = pd.MultiIndex.from_product(remain_col)
# X = [data[remain_col[:8]].values,data[remain_col[8:16]].values,data[remain_col[16:24]].values]
# X_test = [data_test[remain_col[:8]].values,data_test[remain_col[8:16]].values,data_test[remain_col[16:24]].values]
# Y_test = data_test['Label'].values
# Y = data['Label'].values
############################################################################################
##########################::|::ASSIGN DATA::|::################################################
X = [data[fe[:10]].values,data[fe[10:20]].values,data[fe[20:30]].values]
X_test = [data_test[fe[:10]].values,data_test[fe[10:20]].values,data_test[fe[20:30]].values]
Y_test = data_test['Label'].values
Y = data['Label'].values
###################::|:: STANDARDIZATION ::|::##########
# X[0] = preprocessing.scale(X[0])
# X[1] = preprocessing.scale(X[1])
# X[2] = preprocessing.scale(X[2])
#########################::|::LABEL-ENCODE(M->1 and B->0)::|::#################################
la = LabelEncoder()
Y = la.fit_transform(Y)
Y_test = la.fit_transform(Y_test)
c = 0
########################::|:: TRAIN_TEST_SPLIT ::|:: #######################################
result_table = pd.DataFrame(columns=['round','data_source', '#train_sample','#test_sample','classifier','Correct','criterion','splitter','TP','FP','TN','FN','P','N','precision','recall','f-1','ROC_AUC','accuracy'])
for x,name,x_Test in zip(X,['Mean','Standard_error','Max_value'],X_test):
    for i in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        for cri in ['gini','entropy']:
            for split in ['best','random']:
                x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=i, random_state=0)
                c+=1
                clf = tree.DecisionTreeClassifier(criterion=cri,splitter=split)
                clf.fit(x_train,y_train)
                predict = clf.predict(x_test)
                CM = metrics.confusion_matrix(y_test,predict)
                row = {"round": c,
                       'data_source': name,
                       '#train_sample': x_train.shape[0],
                       '#test_sample': x_test.shape[0],
                       'classifier': 'DecisionTree',
                       'Correct':np.sum(predict == y_test),
                       'criterion' :cri,
                       'splitter' : split,
                       'TP': CM[0,0],
                       'FP': CM[0,1],
                       'TN': CM[1,0],
                       'FN': CM[1,1],
                       'P': CM[0,0] + CM[0,1],
                       'N': CM[1,0] + CM[1,1],
                       'precision':round(metrics.precision_score(y_test,predict),4),
                       'recall':round(metrics.recall_score(y_test,predict),4),
                       'f-1':round(metrics.f1_score(y_test,predict),4),
                       'ROC_AUC': round(metrics.roc_auc_score(y_test, predict), 4),
                      'accuracy': round(metrics.accuracy_score(y_test,predict),4)}
                result_table = result_table.append(row, ignore_index=True)
###########################################################################################
############################::|:: DRAW TREE ::|::#########################################
# clf = tree.DecisionTreeClassifier(criterion='entropy',splitter='random')
# clf.fit(X[2],Y)
# dot_plot = tree.export_graphviz(clf,out_file=None,filled=True,rounded=True,feature_names=features_d,class_names=['M','B'])
# graph = graphviz.Source(dot_plot,format='pdf')
# graph.render('Result/DecisionTree/MaxValue_BEST_Recall(entropy,rendom)(Dimensions-Reduced)')
############################################################################################
########################::|::K-FOLD::|::#######################################################
# result_table = pd.DataFrame(columns=['round','data_source', '#train_sample','#test_sample','#Fold','classifier','Correct','criterion','splitter','TP','FP','TN','FN','P','N','precision','recall','f-1','ROC_AUC','accuracy'])
# for x,name,x_Test in zip(X,['Mean','Standard_error','Max_value'],X_test):
#     for i in [2,5,8,10,15]:
#         for cri in ['gini','entropy']:
#             for split in ['best','random']:
#                 kf = KFold(n_splits=i)
#                 for train_index, test_index in kf.split(x):
#                     x_train, x_test = x[train_index], x[test_index]
#                     y_train , y_test = Y[train_index],Y[test_index]
#                     c+=1
#                     clf = tree.DecisionTreeClassifier(criterion=cri,splitter=split)
#                     clf.fit(x_train,y_train)
#                     predict = clf.predict(x_test)
#                     CM = metrics.confusion_matrix(y_test,predict)
#                     row = {"round": c,
#                            'data_source': name,
#                            '#train_sample': x_train.shape[0],
#                            '#test_sample': x_test.shape[0],
#                            '#Fold':i,
#                            'classifier': 'DecisionTree',
#                            'Correct':np.sum(predict == y_test),
#                            'criterion' :cri,
#                            'splitter' : split,
#                            'TP': CM[0,0],
#                            'FP': CM[0,1],
#                            'TN': CM[1,0],
#                            'FN': CM[1,1],
#                            'P': CM[0,0] + CM[0,1],
#                            'N': CM[1,0] + CM[1,1],
#                            'precision':round(metrics.precision_score(y_test,predict),4),
#                            'recall':round(metrics.recall_score(y_test,predict),4),
#                            'f-1':round(metrics.f1_score(y_test,predict),4),
#                            'ROC_AUC': round(metrics.roc_auc_score(y_test, predict), 4),
#                           'accuracy': round(metrics.accuracy_score(y_test,predict),4)}
#                     result_table = result_table.append(row, ignore_index=True)
##########################################################################################
########################::|:: CROSS_VALIDATION ::|::######################################
# def tn(y_true, y_pred):
#     return metrics.confusion_matrix(y_true, y_pred)[0, 0]
# def fp(y_true, y_pred):
#     return metrics.confusion_matrix(y_true, y_pred)[0, 1]
# def fn(y_true, y_pred):
#     return metrics.confusion_matrix(y_true, y_pred)[1, 0]
# def tp(y_true, y_pred):
#     return metrics.confusion_matrix(y_true, y_pred)[1, 1]
# scoring = {'TP': metrics.make_scorer(tp),
#            'FP': metrics.make_scorer(fp),
#            'TN': metrics.make_scorer(tn),
#            'FN': metrics.make_scorer(fn),
#            'precision':'precision',
#            'recll':'recall',
#            'f-1':'f1',
#             'roc_auc':'roc_auc',
#            'accuracy':'accuracy'}
# result_table = pd.DataFrame(columns=['round','data_source','#Fold','classifier','criterion','splitter','TP','FP','TN','FN','P','N','precision','recall','f-1','ROC_AUC','accuracy'])
# for x,name in zip(X,['Mean','Standard_error','Max_value']):
#     for k in [2,5,8,10,15]:
#         for cri in ['gini','entropy']:
#             for split in ['best','random']:
#                 clf = tree.DecisionTreeClassifier(criterion=cri,splitter=split)
#                 cv_res = cross_validate(clf.fit(x, Y), x, Y, cv=k, scoring=scoring)
#                 for i in range(k):
#                     c += 1
#                     row = {"round": c,
#                            'data_source': name,
#                            '#Fold':k,
#                            'classifier': 'DecisionTree',
#                            'criterion' :cri,
#                            'splitter' : split,
#                            'TP': cv_res['test_TP'][i],
#                            'FP': cv_res['test_FP'][i],
#                            'TN': cv_res['test_TN'][i],
#                            'FN': cv_res['test_FN'][i],
#                            'P': cv_res['test_TP'][i] + cv_res['test_FP'][i],
#                            'N': cv_res['test_TN'][i] + cv_res['test_FN'][i],
#                            'precision':round(cv_res['test_precision'][i],4),
#                            'recall':round(cv_res['test_recll'][i],4),
#                            'f-1':round(cv_res['test_f-1'][i],4),
#                            'ROC_AUC': round(cv_res['test_roc_auc'][i], 4),
#                           'accuracy': round(cv_res['test_accuracy'][i],4)}
#                     result_table = result_table.append(row, ignore_index=True)
#############################################################################################
########################::|:: SAVE RESULTS ::|::#############################################
result_table.to_excel('TESTRe/1.xlsx',sheet_name='Result',index=False)