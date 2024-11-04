# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: harbilbediagabaneres
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn import tree
# from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier

# import tensorflow as tf


import warnings
warnings.filterwarnings('ignore')


# FUNCIONES -----------------------------------------------------------------------------#

def linear_reg(X_train, y_train, X_test, y_test, data_type):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    y_lr_train_pred = lr.predict(X_train)
    y_lr_test_pred = lr.predict(X_test)
    
    
    K_r2 = len(lr.get_params())
    n_r2 = len(y_train)
    
    r2_train = r2_score(y_lr_train_pred, y_train)
    r2_test = r2_score(y_lr_test_pred, y_test)
    adjr2_train = 1-((1-r2_train)*(n_r2-1)/(n_r2-K_r2-1))
    adjr2_test = 1-((1-r2_test)*(n_r2-1)/(n_r2-K_r2-1))
    
    # lr.fit(y_train, y_test)
    print('X_train shape',X_train.shape)
    print('r2_score_train',r2_train )
    print('r2_score_test',r2_test )
    print('adjr2_train', adjr2_train)
    print('adjr2_test', adjr2_test)

    # print('intercept', lr.intercept_)
    # print('coeficients', lr.coef_)
    
    df=pd.DataFrame(columns=['y_train','y_train_pred', 'y_test','y_test_pred'])
    df['y_train'] = y_train.values
    df['y_train_pred'] = y_lr_train_pred
    df.iloc[:len(y_test),1] = y_test.values
    df.iloc[:len(y_lr_test_pred),3] = y_lr_test_pred
    
    df.to_csv(data_type+'results_train.csv', index=False, sep=';')
    
    plt.plot(y_train, y_lr_train_pred, 'ro')
    plt.title(data_type)
    plt.xlabel('y_train'+data_type)
    plt.ylabel('y_train_pred'+data_type)
    plt.plot(y_train, y_train)

    plt.savefig(data_type+'lr', dpi=100)

    return lr,  y_lr_train_pred,  y_lr_test_pred



def lda_model(X_train, y_train, X_test, y_test, data_type):
    lda = LDA(n_components=1)
    
    lda.fit(X_train, y_train)
    y_train_pred = lda.predict(X_train)
    y_test_pred = lda.predict(X_test)
    y_train_pred_proba = lda.predict_proba(X_train)
    y_test_pred_proba = lda.predict_proba(X_test)
    
    
    cm_train = confusion_matrix(y_train_pred, y_train)
    print(cm_train)
    print('Accuracy in train set ' + str(accuracy_score(y_train, y_train_pred)))
    
    
    cm_test = confusion_matrix(y_test_pred, y_test)
    print(cm_test)
    print('Accuracy in test set ' + str(accuracy_score(y_test, y_test_pred)))

    
    #y_train = lda.transform(X_train)
    #y_test = lda.transform(X_test)

    df_results_train = pd.DataFrame(columns=['y_train','y_train_pred', 'y_train_pred_proba'])
    df_results_train['y_train'] = y_train
    df_results_train['y_train_pred'] = y_train_pred
    df_results_train['y_train_pred_proba'] = y_train_pred_proba

    df_results_train.to_csv(data_type+'results_train.csv', index=False, sep=';')

    df_results_test = pd.DataFrame(columns=['y_test','y_test_pred', 'y_test_pred_proba'])
    df_results_test['y_test'] = y_test
    df_results_test['y_test_pred'] = y_test_pred
    df_results_test['y_test_pred_proba'] = y_test_pred_proba

    df_results_test.to_csv(data_type+'results_test.csv', index=False, sep=';')
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    accuracy = (tn+tp)/(tn+tp+fn+fp)

    print('Sp(%)=', specificity,'Sn(%)=' , sensitivity,'Ac(%)=' , accuracy)

    return lda,  y_train_pred,  y_test_pred
    
def qda_model(X_train, y_train, X_test, y_test, data_type):
    qda = QDA()
    
    qda.fit(X_train, y_train)
    y_train_pred = qda.predict(X_train)
    y_test_pred = qda.predict(X_test)
    y_test_pred_proba = qda.predict_proba(X_test)
    #y_train_pred_proba = qda.predict_proba(X_train)

    cm_train = confusion_matrix(y_train_pred, y_train)
    print(cm_train)
    print('Accuracy in train set ' + str(accuracy_score(y_train, y_train_pred)))
    
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm_train,display_labels=qda.classes_)
    # disp.plot()

    cm_test = confusion_matrix(y_test_pred, y_test)
    print(cm_test)
    print('Accuracy in test set ' + str(accuracy_score(y_test, y_test_pred)))

    df_results_train = pd.DataFrame(columns=['y_train','y_train_pred', 'y_train_pred_proba'])
    df_results_train['y_train'] = y_train
    df_results_train['y_qda_train_pred'] = y_train_pred
    #df_results_train['y_qda_train_pred_proba'] = y_train_pred_proba

    df_results_train.to_csv(data_type+'results_train.csv', index=False, sep=';')

    df_results_test = pd.DataFrame(columns=['y_test','y_test_pred', 'y_test_pred_proba'])
    df_results_test['y_test'] = y_test
    df_results_test['y_test_pred'] = y_test_pred
    df_results_test['y_test_pred_proba'] = y_test_pred_proba

    df_results_test.to_csv(data_type+'results_test.csv', index=False, sep=';')

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    accuracy = (tn+tp)/(tn+tp+fn+fp)

    print('Sp(%)=', specificity,'Sn(%)=' , sensitivity,'Ac(%)=' , accuracy)
    
    return qda,  y_train_pred,  y_test_pred

def dtc_model(X_train, y_train, X_test, y_test, data_type):
    probability_arr = []
    entropy_arr = []
    gini_arr = []
    info_arr = []

    for col in X_train:
        probability = X_train[col].value_counts(normalize=True)
        #print(col, 'probability')
        probability_arr.append(probability)
        #print(probability)

        entropy = -1 * np.sum(np.log2(probability) * probability)
        #print(col, 'entropy:', entropy)
        entropy_arr.append(entropy)

        gini_index = 1 - np.sum(np.square(probability))
        #print(col, 'gini_index:', gini_index)
        gini_arr.append(gini_index)

        info_gain = np.log2(probability) * probability
        #print(col, 'information gain:', info_gain)
        info_arr.append(info_gain)

    # corrmat = X_train.corr()
    # sns.heatmap(corrmat, vmax=0.8, square=True)

    dtc = tree.DecisionTreeClassifier(max_leaf_nodes=50).fit(X_train, y_train)
    print(dtc.score(X_train,y_train))

    y_train_pred = dtc.predict(X_train)
    y_test_pred = dtc.predict(X_test)
    #y_dtc_train_pred_proba = dtc.predict_proba(X_train,check_input=False)
    #y_dtc_test_pred_proba = dtc.predict_proba(X_test,check_input=False)

    plt.figure(figsize=(28,14))  # set plot size (denoted in inches)
    tree.plot_tree(dtc, fontsize=6)
    plt.savefig('dtc', dpi=100)

    df_results_train = pd.DataFrame(columns=['y_train','y_train_pred']) #, 'y_train_pred_proba'])
    df_results_train['y_train'] = y_train
    df_results_train['y_train_pred'] = y_train_pred
    #df_results_train['y_train_pred_proba'] = y_train_pred_proba

    df_results_train.to_csv(data_type+'results_train.csv', index=False, sep=';')

    df_results_test = pd.DataFrame(columns=['y_test','y_test_pred']) #, 'y_test_pred_proba'])
    df_results_test['y_test'] = y_test
    df_results_test['y_test_pred'] = y_test_pred
    #df_results_test['y_test_pred_proba'] = y_test_pred_proba

    df_results_test.to_csv(data_type+'results_test.csv', index=False, sep=';')

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    accuracy = (tn+tp)/(tn+tp+fn+fp)

    print('Sp(%)=', specificity,'Sn(%)=' , sensitivity,'Ac(%)=' , accuracy)

    return dtc,  y_train_pred,  y_test_pred

def svm_model(X_train, y_train, X_test, y_test, data_type):

    svm_model = svm.SVC(probability=True)
    svm_model.fit(X_train, y_train)

    y_test_pred = svm_model.predict(X_test)
    y_train_pred = svm_model.predict(X_train)   
    y_test_pred_proba = svm_model.predict_proba(X_test)
    y_train_pred_proba = svm_model.predict_proba(X_train) 

    df_results_test = pd.DataFrame(columns=['y_test','y_test_pred','y_test_pred_proba'])
    df_results_test['y_test'] = y_test
    df_results_test['y_test_pred'] = y_test_pred
    df_results_test['y_test_pred_proba'] = y_test_pred_proba

    df_results_test.to_csv(data_type+'results_train.csv', index=False, sep=';')

    df_results_train = pd.DataFrame(columns=['y_train','y_train_pred','y_train_pred_proba'])
    df_results_train['y_train'] = y_train
    df_results_train['y_train_pred'] = y_train_pred
    df_results_train['y_train_pred_proba'] = y_train_pred_proba

    df_results_train.to_csv(data_type+'results_test.csv', index=False, sep=';')

    cm_train = confusion_matrix(y_train_pred, y_train)
    print(cm_train)
    print('Accuracy in train set ' + str(accuracy_score(y_train, y_train_pred)))
    
    
    cm_test = confusion_matrix(y_test_pred, y_test)
    print(cm_test)
    print('Accuracy in test set ' + str(accuracy_score(y_test, y_test_pred)))

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    accuracy = (tn+tp)/(tn+tp+fn+fp)

    print('Sp(%)=', specificity,'Sn(%)=' , sensitivity,'Ac(%)=' , accuracy)

    return svm,  y_train_pred,  y_test_pred

def MLPC_model(X_train, y_train, X_test, y_test, data_type):
    mlpc_model = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='adam',learning_rate='adaptive', max_iter=1000)

    mlpc_model.fit(X_train, y_train)

    y_pred = mlpc_model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    accuracy = (tn+tp)/(tn+tp+fn+fp)

    print('Sp(%)=', specificity,'Sn(%)=' , sensitivity,'Ac(%)=' , accuracy)

    y_test_pred = mlpc_model.predict(X_test)
    y_train_pred = mlpc_model.predict(X_train)   
    y_test_pred_proba = mlpc_model.predict_proba(X_test)
    y_train_pred_proba = mlpc_model.predict_proba(X_train) 

    df_results_test = pd.DataFrame(columns=['y_test','y_test_pred','y_test_pred_proba'])
    df_results_test['y_test'] = y_test
    df_results_test['y_test_pred'] = y_test_pred
    df_results_test['y_test_pred_proba'] = y_test_pred_proba.tolist()

    df_results_test.to_csv(data_type+'results_test.csv', index=True, sep=';')

    df_results_train = pd.DataFrame(columns=['y_train','y_train_pred','y_train_pred_proba'])
    df_results_train['y_train'] = y_train
    df_results_train['y_train_pred'] = y_train_pred
    df_results_train['y_train_pred_proba'] = y_train_pred_proba.tolist()

    df_results_train.to_csv(data_type+'results_train.csv', index=True, sep=';')

    return mlpc_model,  y_train_pred,  y_test_pred

# ENTRENAMIENTO DE DIFERENTES MODELOS -----------------------------------------------------------------------------#

PATH = Path(__file__).parents[0]# / Path('data/')

#Abrir csv
X_var_data = pd.read_csv (str(PATH / Path('X_var_data.csv')), sep=';')
X_var_data_norm = pd.read_csv (str(PATH / Path('X_var_data_norm.csv')), sep=';')
X_var_data_stand = pd.read_csv (str(PATH / Path('X_var_data_stand.csv')), sep=';')
MA_hydrogel_work = pd.read_csv (str(PATH / Path('MA_hydrogel_work.csv')), sep=';')
y_var = pd.read_csv (str(PATH / Path('y_var.csv')), sep=';')


#CÓDIGO PARA EJECUTAR
execute_LR = False
execute_LDA = True
execute_QDA = False
execute_DTC = False
execute_SVM = False
execute_MLPC = False


# REGRESIÓN LINEAL -----------------------------------------------------------------------------#
if execute_LR:
    # MODELO RAW
    print('\n LR Ajuste inicial - RAW')
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_var_data, y_var, test_size=0.2, random_state=42)
    lr,  y_lr_train_pred,  y_lr_test_pred= linear_reg(X_train_raw, y_train_raw, X_test_raw, y_test_raw, 'lr - raw')

    model = SelectFromModel(lr, prefit=True)
    X_train_raw_new = model.transform(X_train_raw)
    X_test_raw_new = model.transform(X_test_raw)

    print('\n LR Ajuste new - RAW')
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X_var_data_norm, MA_hydrogel_work['Value_norm'], test_size=0.2, random_state=42)
    lr_new,  y_lr_train_pred_new,  y_lr_test_pred_new= linear_reg(X_train_raw_new, y_train_raw, X_test_raw_new, y_test_raw, 'lr - raw_new')

    # MODELO NORM
    print('\n LRAjuste inicial - NORM')
    X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X_var_data_stand, MA_hydrogel_work['Value_std'], test_size=0.2, random_state=42)
    lr_norm,  y_lr_train_pred_norm,  y_lr_test_pred_norm = linear_reg(X_train_norm, y_train_norm, X_test_norm, y_test_norm, 'lr - norm')

    model = SelectFromModel(lr_norm, prefit=True)
    X_train_norm_new = model.transform(X_train_norm)
    X_test_norm_new = model.transform(X_test_norm)

    print('\n LR Ajuste new - NORM')
    lr_norm_new,  y_lr_train_pred_norm_new,  y_lr_test_pred_norm_new= linear_reg(X_train_norm_new, y_train_norm, X_test_norm_new, y_test_norm, 'lr - norm_new')

    #MODELO STD
    print('\n LR Ajuste inicial - STD')
    lr_std,  y_lr_train_pred_std,  y_lr_test_pred_std= linear_reg(X_train_std, y_train_std, X_test_std, y_test_std, 'lr - std')

    model = SelectFromModel(lr_std, prefit=True)
    X_train_std_new = model.transform(X_train_std)
    X_test_std_new = model.transform(X_test_std)

    print('\n LR Ajuste new - STD')
    lr_std_new,  y_lr_train_pred_std_new,  y_lr_test_pred_std_new= linear_reg(X_train_std_new, y_train_std, X_test_std_new, y_test_std, 'std_new')


# LINEAR DISCRIMINANT ANALYSIS -----------------------------------------------------------------------------#
if execute_LDA:
    # MODELO RAW
    print('\n LDA - RAW')
    lda_X_train_raw, lda_X_test_raw, lda_y_train_raw, lda_y_test_raw = train_test_split(X_var_data, MA_hydrogel_work['Value_LDA'], test_size=0.3, random_state = 42)
    lda,  y_lda_train_pred,  y_lda_test_pred = lda_model(lda_X_train_raw, lda_y_train_raw, lda_X_test_raw, lda_y_test_raw, 'lda_raw')

    # MODELO NORM
    print('\n LDA - NORM')
    lda_X_train_norm, lda_X_test_norm, lda_y_train_norm, lda_y_test_norm = train_test_split(X_var_data_norm, MA_hydrogel_work['Value_LDA'], test_size=0.3, random_state = 42)
    lda,  y_lda_train_pred,  y_lda_test_pred = lda_model(lda_X_train_norm, lda_y_train_norm, lda_X_test_norm, lda_y_test_norm, 'lda - norm')

    # MODELO STD
    print('\n LDA - STD')
    lda_X_train_std, lda_X_test_std, lda_y_train_std, lda_y_test_std = train_test_split(X_var_data_stand, MA_hydrogel_work['Value_LDA'], test_size=0.3, random_state = 42)
    lda,  y_lda_train_pred,  y_lda_test_pred = lda_model(lda_X_train_std, lda_y_train_std, lda_X_test_std, lda_y_test_std, 'lda - std')

# QUADRATIC DISCRIMINANT ANALYSIS -----------------------------------------------------------------------------#
if execute_QDA:
    # MODELO RAW
    print('\n QDA - RAW')
    qda_X_train_raw, qda_X_test_raw, qda_y_train_raw, qda_y_test_raw = train_test_split(X_var_data, MA_hydrogel_work['Value_LDA'], test_size=0.3, random_state = 42)
    qda,  y_qda_train_pred,  y_qda_test_pred = qda_model(qda_X_train_raw, qda_y_train_raw, qda_X_test_raw, qda_y_test_raw, 'qda_raw')

    # MODELO NORM
    print('\n QDA - NORM')
    qda_X_train_norm, qda_X_test_norm, qda_y_train_norm, qda_y_test_norm = train_test_split(X_var_data_norm, MA_hydrogel_work['Value_LDA'], test_size=0.3, random_state = 42)
    qda,  y_qda_train_pred,  y_qda_test_pred = qda_model(qda_X_train_norm, qda_y_train_norm, qda_X_test_norm, qda_y_test_norm, 'qda - norm')

    # MODELO STD
    print('\n QDA - STD')
    qda_X_train_std, qda_X_test_std, qda_y_train_std, qda_y_test_std = train_test_split(X_var_data_stand, MA_hydrogel_work['Value_LDA'], test_size=0.3, random_state = 42)
    qda,  y_qda_train_pred,  y_qda_test_pred = qda_model(qda_X_train_std, qda_y_train_std, qda_X_test_std, qda_y_test_std, 'qda - std')


# DECISSION TREE CLASIFIER -----------------------------------------------------------------------------#
if execute_DTC:
    # MODELO RAW
    print('\n DTC - RAW')
    df_tree_raw = X_var_data.drop(['Unnamed: 0'], axis=1)
    df_tree_raw['Class'] = MA_hydrogel_work['Value_LDA']
    df_tree_noclass_raw = df_tree_raw.drop(columns=['Class'])

    dtc_X_train, dtc_X_test, dtc_y_train, dtc_y_test = train_test_split(df_tree_noclass_raw, df_tree_raw['Class'], test_size=0.3, random_state = 42)
    dtc,  y_dtc_train_pred,  y_dtc_test_pred = dtc_model(dtc_X_train, dtc_y_train, dtc_X_test, dtc_y_test, 'dtc')

    # MODELO NORM
    print('\n DTC - NORM')
    df_tree_norm = X_var_data_norm.drop(['Unnamed: 0'], axis=1)
    df_tree_norm['Class'] = MA_hydrogel_work['Value_LDA']
    df_tree_noclass_norm = df_tree_norm.drop(columns=['Class'])

    dtc_X_train, dtc_X_test, dtc_y_train, dtc_y_test = train_test_split(df_tree_noclass_norm, df_tree_norm['Class'], test_size=0.3, random_state = 42)
    dtc,  y_dtc_train_pred,  y_dtc_test_pred = dtc_model(dtc_X_train, dtc_y_train, dtc_X_test, dtc_y_test, 'dtc')

    # MODELO STD
    print('\n DTC - STD')
    df_tree_stand = X_var_data_stand.drop(['Unnamed: 0'], axis=1)
    df_tree_stand['Class'] = MA_hydrogel_work['Value_LDA']
    df_tree_noclass_stand = df_tree_stand.drop(columns=['Class'])

    dtc_X_train, dtc_X_test, dtc_y_train, dtc_y_test = train_test_split(df_tree_noclass_stand, df_tree_stand['Class'], test_size=0.3, random_state = 42)
    dtc,  y_dtc_train_pred,  y_dtc_test_pred = dtc_model(dtc_X_train, dtc_y_train, dtc_X_test, dtc_y_test, 'dtc')

# SUPPORT VECTOR MACHINE -----------------------------------------------------------------------------#
if execute_SVM:
    # MODELO RAW
    print('\n SVM - RAW')
    svm_X_train, svm_X_test, svm_y_train, svm_y_test = train_test_split(X_var_data, MA_hydrogel_work['Value_LDA'], test_size=0.3, random_state = 42)
    svm,  y_svm_train_pred,  y_svm_test_pred = svm_model(svm_X_train, svm_y_train, svm_X_test, svm_y_test, 'svm')

    # MODELO NORM
    print('\n SVM - NORM')
    svm_X_train_norm, svm_X_test_norm, svm_y_train_norm, svm_y_test_norm = train_test_split(X_var_data_norm, MA_hydrogel_work['Value_LDA'], test_size=0.3, random_state = 42)
    svm,  y_svm_train_pred,  y_svm_test_pred = svm_model(svm_X_train_norm, svm_y_train_norm, svm_X_test_norm, svm_y_test_norm, 'svm - norm')

    # MODELO STD
    print('\n SVM -STD')
    svm_X_train_std, svm_X_test_std, svm_y_train_std, svm_y_test_std = train_test_split(X_var_data_stand, MA_hydrogel_work['Value_LDA'], test_size=0.3, random_state = 42)
    svm,  y_svm_train_pred,  y_svm_test_pred = svm_model(svm_X_train_std, svm_y_train_std, svm_X_test_std, svm_y_test_std, 'svm - std')

# NEURAL NETWORKS -----------------------------------------------------------------------------#

if execute_MLPC:
    nn_X_train, nn_X_test_std, nn_y_train_std, nn_y_test_std = train_test_split(X_var_data_stand, MA_hydrogel_work['Value_LDA'], test_size=0.3, random_state = 42)

    print('\n MLPC -STD')
    mlpc,  y_mlpc_train_pred,  y_mlpc_test_pred = MLPC_model(nn_X_train, nn_y_train_std, nn_X_test_std, nn_y_test_std, 'mlpc - std')
