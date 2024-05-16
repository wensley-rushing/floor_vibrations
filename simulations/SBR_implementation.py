import numpy as np
import pandas as pd
from itertools import product

from data_simulation import damping


# extract CLT list from excel

ES_RMS_table_full = pd.read_excel('Vloertrilling_prEN_SBR_EC5_17102023.xlsx', sheet_name = 'ES_RMS_90_full')
ES_RMS_table = ES_RMS_table_full[ES_RMS_table_full['damping'] == damping]
ES_RMS_table.set_index('f', inplace = True)

print(ES_RMS_table)