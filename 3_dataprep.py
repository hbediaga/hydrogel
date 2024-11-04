# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: harbilbediagabaneres
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# PRE-PROCESAMIENTO DE DATOS -----------------------------------------------------------------------------#

PATH = Path(__file__).parents[0]# / Path('data/')

#Abrir csv
MA_hydrogel = pd.read_csv (str(PATH / Path('MA_minitable_alva.csv')), sep=';')

# PRE-PROCESAMIENTO DE LA VARIABLE Y

#Se crea un dataframe con los c0 y los values
MA_hydrogel_work = MA_hydrogel[['c0=Outcome', 'Value']].copy()
MA_hydrogel_work = MA_hydrogel_work.astype({'Value':float})
y_var = MA_hydrogel_work['Value']

#se obtiene una lista de todos los c0 diferentes y se calcula media, sd, min y max para cada uno de ellos
y_list_outcomes = MA_hydrogel_work['c0=Outcome'].unique().tolist()
avg_outcomes = MA_hydrogel_work.groupby('c0=Outcome', as_index=False).mean()
std_outcomes = MA_hydrogel_work.groupby('c0=Outcome', as_index=False).std()
min_outcomes = MA_hydrogel_work.groupby('c0=Outcome', as_index=False).min()
max_outcomes = MA_hydrogel_work.groupby('c0=Outcome', as_index=False).max()

#Se crean listas vacías de los valores observados para poder hacer diferentes modelos
MA_hydrogel_work['Value_norm'] = None #valor normalizado: se reescala entre [0,1]
MA_hydrogel_work['Value_std'] = None #valor estandarizado: la media se pone en 0 y se reescala en +-1 unidad de desvest
MA_hydrogel_work['Value_LDA'] = None 

#rellenamos las listas con un bucle. Para cada valor, realizar una operación dependiendo de los valores de cada outcome
#para calcular el valor normalizado y estandarizado de las variables se calculan los valores min, max, avg y sd
for outcome in y_list_outcomes:
    value_avg = avg_outcomes.loc[avg_outcomes['c0=Outcome'] == outcome, 'Value'].values[0]
    value_std = std_outcomes.loc[std_outcomes['c0=Outcome'] == outcome, 'Value'].values[0]
    value_min = (min_outcomes.loc[min_outcomes['c0=Outcome'] == outcome, 'Value'].values[0])*0.5
    value_max = (max_outcomes.loc[max_outcomes['c0=Outcome'] == outcome, 'Value'].values[0])*1.5
    
#Control. Max y Min si son iguales en alguna c0, nohay división entre 0 
# Se calculan los valores para los normalizados y estandarizados y se van metiendo en las columnas de la tabla que se ha creado vacía   
    if value_min != value_max:
        mask = MA_hydrogel_work['c0=Outcome'] == outcome
        MA_hydrogel_work.loc[mask,'Value_norm'] = (MA_hydrogel_work['Value'] -  value_min)/ (value_max  -  value_min)
        MA_hydrogel_work.loc[mask,'Value_std'] = (MA_hydrogel_work['Value'] -  value_avg)/ (value_std)
    else:
        MA_hydrogel_work.loc[mask,'Value_norm'] = 0
        MA_hydrogel_work.loc[mask,'Value_std'] = 0

#Para rellenar la última columna de valores de la tabla, que serían los valores clasificados en clases, se carga una tabla con los límites para ser clasificado en cada grupo    
df_lda_class = pd.read_csv (str(PATH / Path('df_lda_class.csv')), sep=';')

#se crea un bucle con el que se analiza cada valor
for outcome in y_list_outcomes: #primero se obtiene el valor de cada límite para cada uno de los outcome
    lim0 = df_lda_class.loc[df_lda_class['c0=Outcome']==outcome,'lim0'].values[0]
    lim1 = df_lda_class.loc[df_lda_class['c0=Outcome']==outcome,'lim1'].values[0]
    
    #se crean varias mask para que la función se simplifique
    mask_outcome = MA_hydrogel_work['c0=Outcome']==outcome
    
    mask_dentro_arriba = MA_hydrogel_work.loc[mask_outcome,'Value']<=lim1
    mask_dentro_abajo = MA_hydrogel_work.loc[mask_outcome,'Value']>=lim0

    mask_fuera_arriba = MA_hydrogel_work.loc[mask_outcome,'Value']>lim1
    mask_fuera_abajo = MA_hydrogel_work.loc[mask_outcome,'Value']<lim0 
    
    #por último se crean las máscaras para cada una de las clases
    mask_class_0 = mask_fuera_arriba
    mask_class_00 = mask_fuera_abajo
    mask_class_1 = mask_dentro_arriba & mask_dentro_abajo

#Y se mete el valor de cada elemento en la variable 'Value' clasificada en la variable 'Value_LDA'   
    MA_hydrogel_work.loc[mask_outcome & mask_class_0,'Value_LDA'] = '0'
    MA_hydrogel_work.loc[mask_outcome & mask_class_00,'Value_LDA'] = '0'
    MA_hydrogel_work.loc[mask_outcome & mask_class_1,'Value_LDA'] = '1'

# PRE-PROCESAMIENTO DE LA VARIABLE X
#se crea una lista de columnas de descriptores                     
X_cols = []
for column in MA_hydrogel:
    if column in ('Ref',
                  'DOI',
                  'Value',
                  'Crosslinker',
                  'CL conc (mM)',
                  'c0=Outcome',
                  'c1=Extrusion P (kPa)',
                  'c2=Extrusion speed (mm/s)',
                  'c3=Nozzle',
                  'c4=Nozzle inner diam (um)',
                  'c5=Layers printed',
                  'c6=Poly temp (ºC)',
                  'c7=Syringe temp (ºC)',
                  'c8=Platform temp (ºC)',
                  'c9=EtOH') or column[:4]=='Comp' :
        continue
    X_cols.append(column)

#se obtiene la lista de valores     
X_var = MA_hydrogel[X_cols].values
X_var[np.isnan(X_var)] = 0
X_var_data = pd.DataFrame(data=X_var, columns=X_cols)

#DataFrame con los valores de las variables ESTANDARIZADOS
X_var_data_stand = pd.DataFrame(data=X_var_data.values, columns=X_cols)

for column in X_var_data_stand:
    mean_value = X_var_data_stand[column].mean()
    std_value = X_var_data_stand[column].std()
    
    X_var_data_stand[column]=((X_var_data_stand[column]-mean_value)/std_value)

#DataFrame con los valores de las variables ESTANDARIZADOS
X_var_data_norm = pd.DataFrame(data=X_var_data.values, columns=X_cols)

for column in X_var_data_norm:
    min_value = X_var_data_norm[column].min()
    max_value = X_var_data_norm[column].max()
     
    arriba = X_var_data_norm[column]-min_value
    abajo = max_value-min_value
    X_var_data_norm[column] = arriba / abajo


X_var_data.to_csv('X_var_data.csv', index=False, sep=';')
X_var_data_norm.to_csv('X_var_data_norm.csv', index=False, sep=';')
X_var_data_stand.to_csv('X_var_data_stand.csv', index=False, sep=';')
MA_hydrogel_work.to_csv('MA_hydrogel_work.csv', index=False, sep=';')
y_var.to_csv('y_var.csv', index=False, sep=';')