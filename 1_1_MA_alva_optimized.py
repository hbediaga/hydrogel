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


######### CODIGO #########

# Definición del PATH
PATH = Path(__file__).parents[0] / Path('data/')

# Lectura de la base de datos de hidrogeles
DB_hydrogel = pd.read_csv(str(PATH / Path('DB.csv')), sep=';')
DB_hydrogel['Comp01'] = DB_hydrogel['Comp01'].str.lower()
DB_hydrogel['Comp02'] = DB_hydrogel['Comp02'].str.lower()

# Lectura de los descriptores
Desc = pd.read_csv(str(PATH / Path('alva_desc.csv')), sep=';')
Desc.replace('#N/D', 0, inplace=True)
Desc['filename'] = Desc['filename'].str.lower()

# Renombrar las columnas de los descriptores
Desc.rename(columns={"filename": "Comp01"}, inplace=True)

# Crear un diccionario para obtener los valores únicos para cada condición
dict_var = {}

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

for column in condition_columns:
    dict_var[column] = DB_hydrogel[column].unique().tolist()

# Realizar el merge de los descriptores con la base de datos de hidrogeles
buscarv_DB = DB_hydrogel.copy()

for comp_col in ['Comp01']:
    buscarv_df = buscarv_DB.merge(Desc, left_on=comp_col, right_on='Comp01', how='left', suffixes=('_' + comp_col, ''))
    buscarv_DB = buscarv_df.drop(columns=[ 'Comp02', 'Comp02 (w/v%)'])

buscarv_DB = buscarv_DB.replace('-', None)

# Multiplicar el valor de cada descriptor por la composición y sumar los valores
for column in Desc.columns:
    if column in ('Comp01', 'SMILES'):
        continue
    for comp_col in ['Comp01']:
        buscarv_DB.loc[buscarv_DB[column] == '-', column] = None
        buscarv_DB[column] = buscarv_DB[column].astype(float) * (
                buscarv_DB[comp_col + ' (w/v%)'].astype(float) / 100)

    # buscarv_DB[column] = buscarv_DB[column + '_Comp01'].fillna(0) + buscarv_DB[column + '_Comp02'].fillna(0)
    # buscarv_DB.drop(columns=[column + '_Comp01', column + '_Comp02'], inplace=True)

    buscarv_DB[column] = buscarv_DB[column].fillna(0)
    buscarv_DB.drop(columns=[column])


# Crear una tabla con los promedios de cada entrada de cada condición
func_ref = buscarv_DB.groupby('c0=Outcome')['Value'].mean().reset_index()
func_ref = func_ref.rename(columns={'Value': 'func_ref'})

# Crear una tabla con los promedios de cada condición
desc_list = Desc.columns.tolist()
desc_list.remove('SMILES')

average_table = buscarv_DB.groupby('c0=Outcome').mean()

avg_results = pd.DataFrame()

for column in condition_columns:
    col_name, desc = column.split('=')
    if column == 'c0=Outcome':
        continue

    avg_result = buscarv_DB[[column] + desc_list].groupby(column, as_index=False).mean()
    avg_result = avg_result.rename(columns=rename_vars(desc_list, col_name))
    avg_results = pd.concat([avg_results, avg_result], axis=1)

# Si quieres incluir 'c0=Outcome' como una de las columnas en avg_results
avg_c0 = buscarv_DB[['c0=Outcome'] + desc_list].groupby('c0=Outcome', as_index=False).mean()
avg_c0 = avg_c0.rename(columns=rename_vars(desc_list, 'c0'))

avg_results = pd.concat([avg_c0, avg_results], axis=1)

# Resta los valores promedio por cada condición para calcular los MA
for desc in desc_list:
    for col_name in condition_columns:
        if desc == 'Comp01':
            continue

        avg_results[desc + '_' + col_name] = avg_results[desc] - avg_results[desc + '_c0']

# Ahora 'avg_results' contiene los resultados con DataFrames en lugar de diccionarios






avg_results = {}

for column in condition_columns:
    col_name, desc = column.split('=')
    if column in ('Comp01',):
        continue

    avg_result = buscarv_DB[[column] + desc_list].groupby(column, as_index=False).mean()
    avg_result = avg_result.rename(columns=rename_vars(desc_list, col_name))
    avg_results[col_name] = avg_result

# Calcular los MA
MA_table = buscarv_DB.copy()

for col_name, avg_result in avg_results.items():
    MA_table = MA_table.merge(avg_result, 'left', col_name)

for desc in desc_list:
    for col_name in condition_columns:
        MA_table[desc + '_' + col_name] = MA_table[desc] - MA_table[desc + '_' + col_name]

tabla_con_ref = MA_table.merge(func_ref, 'left', 'c0=Outcome')
