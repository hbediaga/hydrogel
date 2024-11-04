#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: harbilbediagabaneres
"""
import pandas as pd
from glob import glob
from pathlib import Path

#función para renombrar las variables. Escribe detrás el número de la condición
def rename_vars(columns,suffix):
    dict_rename = {}
    for column in columns:
        dict_rename[column] = column + '_' + suffix
        
    return dict_rename

PATH = Path(__file__).parents[0] / Path('data/')

DB_hydrogel = pd.read_csv(str(PATH / Path('DB.csv')), sep=';')
DB_hydrogel['Comp01'] = DB_hydrogel['Comp01'].str.lower()
DB_hydrogel['Comp02'] = DB_hydrogel['Comp02'].str.lower()

#Abrimos la base de datos de hidrogeles solo con las condiciones
#DB_hydrogel = pd.read_csv ('/DB.csv', sep=';')

#Se abre dos veces el archivo de compuestos y descriptores. Se va a hacer merge dos veces
Desc1 = pd.read_csv(str(PATH / Path('alva_desc.csv')), sep=';')
Desc2 = pd.read_csv(str(PATH / Path('alva_desc.csv')), sep=';')
Desc1.replace('#N/D',0,inplace=True)
Desc2.replace('#N/D',0,inplace=True)

#Se define una lista de condiciones de la base de datos
condition_columns=[
    'c0=Outcome',
    'c1=Extrusion P (kPa)',
    'c2=Extrusion speed (mm/s)',
    'c3=Nozzle',
    'c4=Nozzle inner diam (um)',
    'c5=Layers printed',
    'c6=Poly temp (ºC)',
    'c7=Syringe temp (ºC)',
    'c8=Platform temp (ºC)',
    'c9=EtOH'
]

#Creamos un diccionario vacío para obtener los valores únicos para cada condición
dict_var = {}

for column in condition_columns:
    dict_var[column] = DB_hydrogel[column].unique().tolist()

#Renombramos las columnas de compuestos para poder hacer el merge en la tabla de descriptores
Desc1.rename(columns={"filename": "Comp01"}, inplace=True)
Desc2.rename(columns={"filename": "Comp02"}, inplace=True)
Desc1['Comp01'] = Desc1['Comp01'].str.lower()
Desc2['Comp02'] = Desc2['Comp02'].str.lower()

#Se buscan los descriptores del primer compuesto de la base de datos de hidrogeles. Se traen estos valores a la izquiera de la tabla. Se renombran acabados en '_x'
buscarv_df1 = DB_hydrogel.merge(Desc1, on='Comp01', how='left')

#A la tabla anterior se le suman los descriptores del segundo compuesto a la derecha. Se renombran con '_y'
buscarv_DB = buscarv_df1.merge(Desc2, on='Comp02', how='left')

buscarv_DB.to_csv("data/buscarv_DB_alva.csv", sep=";", index=False)
buscarv_DB = buscarv_DB.replace('-', None)


l_columns_x = []
l_columns_y = []
#Se multipica el valor de cada descriptor por la composición y se suman los valores
for column in Desc1.columns:
    if column in ('Comp01', 'SMILES'):
        continue
    buscarv_DB.loc[buscarv_DB[column+'_x']=='-',column+'_x'] = None
    buscarv_DB.loc[buscarv_DB[column+'_y']=='-',column+'_y'] = None
    buscarv_DB[column+'_x'] = buscarv_DB[column+'_x'].astype(float) * (buscarv_DB['Comp01 (w/v%)'].astype(float)/100)
    buscarv_DB[column+'_y'] = buscarv_DB[column+'_y'].astype(float) * (buscarv_DB['Comp02 (w/v%)'].astype(float)/100)
    
    headers = list(buscarv_DB.columns)
    l_columns_x.append(column+'_x')
    l_columns_y.append(column+'_y')

    buscarv_DB[column+'_y']=buscarv_DB[column+'_y'].fillna(0)
#Se suman las dos columnas de los descriptores    
    buscarv_DB[column] = buscarv_DB[column+'_x']+buscarv_DB[column+'_y']


#Quita todas las columnas que tengan _x y _y. Se crea una tabla con las condiciones y la suma de los descriptores (ponderados por peso)
buscarv_DB = buscarv_DB.drop(columns=['SMILES_y', 'SMILES_x', 'Comp01'])

buscarv_DB.drop(columns = l_columns_x + l_columns_y, inplace=True)

#Se crea una tabla con los promedios de cada entrada de cada condición
desc_list = Desc1.columns.tolist()
desc_list.remove('SMILES')
desc_list.remove('Comp01')

avg_c0 = buscarv_DB[['c0=Outcome']+desc_list].groupby('c0=Outcome', as_index=False).mean()
avg_c0 = avg_c0.rename(columns = rename_vars(desc_list,'c0'))
avg_c0.to_csv('avg_c0.csv', sep=';')

avg_c1 = buscarv_DB[['c1=Extrusion P (kPa)']+desc_list].groupby('c1=Extrusion P (kPa)', as_index=False).mean()
avg_c1 = avg_c1.rename(columns = rename_vars(desc_list,'c1'))
avg_c1.to_csv('avg_c1.csv', sep=';')

avg_c2 = buscarv_DB[['c2=Extrusion speed (mm/s)']+desc_list].groupby('c2=Extrusion speed (mm/s)', as_index=False).mean()
avg_c2 = avg_c2.rename(columns = rename_vars(desc_list,'c2'))
avg_c2.to_csv('avg_c2.csv', sep=';')

avg_c3 = buscarv_DB[['c3=Nozzle']+desc_list].groupby('c3=Nozzle', as_index=False).mean()
avg_c3 = avg_c3.rename(columns = rename_vars(desc_list,'c3'))
avg_c3.to_csv('avg_c3.csv', sep=';')

avg_c4 = buscarv_DB[['c4=Nozzle inner diam (um)']+desc_list].groupby('c4=Nozzle inner diam (um)', as_index=False).mean()
avg_c4 = avg_c4.rename(columns = rename_vars(desc_list,'c4'))
avg_c4.to_csv('avg_c4.csv', sep=';')

avg_c5 = buscarv_DB[['c5=Layers printed']+desc_list].groupby('c5=Layers printed', as_index=False).mean()
avg_c5 = avg_c5 .rename(columns = rename_vars(desc_list,'c5'))
avg_c5.to_csv('avg_c5.csv', sep=';')

avg_c6 = buscarv_DB[['c6=Poly temp (ºC)']+desc_list].groupby('c6=Poly temp (ºC)', as_index=False).mean()
avg_c6 = avg_c6.rename(columns = rename_vars(desc_list,'c6'))
avg_c6.to_csv('avg_c6.csv', sep=';')

avg_c7 = buscarv_DB[['c7=Syringe temp (ºC)']+desc_list].groupby('c7=Syringe temp (ºC)', as_index=False).mean()
avg_c7 = avg_c7.rename(columns = rename_vars(desc_list,'c7'))
avg_c7.to_csv('avg_c7.csv', sep=';')

avg_c8 = buscarv_DB[['c8=Platform temp (ºC)']+desc_list].groupby('c8=Platform temp (ºC)', as_index=False).mean()
avg_c8 = avg_c8.rename(columns = rename_vars(desc_list,'c8'))
avg_c8.to_csv('avg_c8.csv', sep=';')

avg_c9 = buscarv_DB[['c9=EtOH']+desc_list].groupby('c9=EtOH', as_index=False).mean()
avg_c9 = avg_c9.rename(columns = rename_vars(desc_list,'c9'))
avg_c9.to_csv('avg_c9.csv', sep=';')

func_ref = pd.DataFrame()
func_ref['c0=Outcome'] = buscarv_DB['c0=Outcome'].unique()
promedio = buscarv_DB.groupby('c0=Outcome')['Value'].mean().reset_index()
func_ref = pd.merge(func_ref, promedio, on='c0=Outcome')
func_ref = func_ref.rename(columns={'Value':'func_ref'})
func_ref.to_csv('func_ref.csv', sep=';')

#junta todas las tablas anteriores a la tabla con los descriptores para cada entrada

MA_table = (buscarv_DB.merge(avg_c0, 'left', 'c0=Outcome')
      .merge(avg_c1, 'left', 'c1=Extrusion P (kPa)')
      .merge(avg_c2, 'left', 'c2=Extrusion speed (mm/s)')
      .merge(avg_c3, 'left', 'c3=Nozzle')
      .merge(avg_c4, 'left', 'c4=Nozzle inner diam (um)')
      .merge(avg_c5, 'left', 'c5=Layers printed')
      .merge(avg_c6, 'left', 'c6=Poly temp (ºC)')
      .merge(avg_c7, 'left', 'c7=Syringe temp (ºC)')
      .merge(avg_c8, 'left', 'c8=Platform temp (ºC)')
      .merge(avg_c9, 'left', 'c9=EtOH')  
)


desc_list.remove('Comp01')

#resta la media por cada condición para calcular los MA
for desc in desc_list:
    MA_table[desc+'_c0'] = MA_table[desc] - MA_table[desc+'_c0']
    MA_table[desc+'_c1'] = MA_table[desc] - MA_table[desc+'_c1']
    MA_table[desc+'_c2'] = MA_table[desc] - MA_table[desc+'_c2']
    MA_table[desc+'_c3'] = MA_table[desc] - MA_table[desc+'_c3']
    MA_table[desc+'_c4'] = MA_table[desc] - MA_table[desc+'_c4']
    MA_table[desc+'_c5'] = MA_table[desc] - MA_table[desc+'_c5']
    MA_table[desc+'_c6'] = MA_table[desc] - MA_table[desc+'_c6']
    MA_table[desc+'_c7'] = MA_table[desc] - MA_table[desc+'_c7']
    MA_table[desc+'_c8'] = MA_table[desc] - MA_table[desc+'_c8']
    MA_table[desc+'_c9'] = MA_table[desc] - MA_table[desc+'_c9']

tabla_con_ref = (MA_table.merge(func_ref, 'left', 'c0=Outcome'))

#Guardar archivo en csv

tabla_con_ref.to_csv('MA_table_alva.csv', header=True, index=False)