import numpy as np
import pandas as pd
from itertools import product


# extract CLT list from excel

CLT_table_columns = ['naam', 'merk', 'Nummer', 'lagen', 'dikte', 'gewicht', 'D11', 'D22', 'D44']
CLT_table = pd.read_excel('Vloertrilling_prEN_SBR_EC5_17102023.xlsx', sheet_name = 'Tabel_CLT', usecols = CLT_table_columns)
Derix_CLT_table = CLT_table[CLT_table['merk'] == 'Derix']


# Create database parameters
parameters = {
    'Nummer': {'min': 1, 'max': 67, 'step': 1},
    'floor_width': {'min': 2.7, 'max': 10.8, 'step': 2.7},
    'floor_span': {'min': 3, 'max': 8, 'step': 0.5},
    'permanent_load': {'min': 0, 'max': 300, 'step': 100},
    'variable_load': {'min': 0, 'max': 200, 'step': 100}
}

all_combinations = list(product(*[np.linspace(param['min'], param['max'], int((param['max'] - param['min']) / param['step']) + 1) for param in parameters.values()]))

database = pd.DataFrame(all_combinations, columns = parameters.keys())
database_full = pd.merge(database, Derix_CLT_table, how = 'inner', on = ['Nummer'])

# cull database based on ULS check
bending_strength = 1.15 * 0.8 * 24 / 1.25 # k_sys * k_mod * f_m,xlay,k / gamma_m
shear_strength = 0.8 * 4 / 1.25 # k_mod * f_v,090,ylay,k / gamma_m

def calculate_bending_unity_check(row):
    load = (1.35 * (row['gewicht'] + row['permanent_load']) + 1.5 * row['variable_load']) * 0.0098 # [kN/m]
    moment = load * row['floor_span'] ** 2 / 8 # [kNm]
    section_modulus = 2 * (row['D11'] * 1000 / 12000) / row['dikte'] # m**3
    stress = moment / section_modulus # [kN/m]
    unity_check = stress / bending_strength

    return unity_check

database_full['unity_check_bending'] = database_full.apply(calculate_bending_unity_check, axis = 1)

unity_check_limit = 1
filtered_database_full = database_full[database_full['unity_check_bending'] < unity_check_limit]
