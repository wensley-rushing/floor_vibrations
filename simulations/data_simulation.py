import numpy as np
import pandas as pd
from itertools import product


# extract CLT list from excel

CLT_table_columns = ['naam', 'merk', 'Nummer', 'lagen', 'dikte', 'gewicht', 'D11', 'D22']
CLT_table = pd.read_excel('Vloertrilling_prEN_SBR_EC5_17102023.xlsx', sheet_name='Tabel_CLT', usecols=CLT_table_columns)
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

database = pd.DataFrame(all_combinations, columns=parameters.keys())
database_full = pd.merge(database, Derix_CLT_table, how='inner', on = ['Nummer'])

# cull database based on ULS check
