import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from itertools import product


# extract CLT list from excel

CLT_table_columns = ['naam', 'merk', 'Nummer', 'lagen', 'dikte', 'gewicht', 'D11', 'D22']
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

# ANALYTICAL FORMULATIONS FOR MODAL PROPERTIES

# set definition floors
span_type = 'two-way'

def define_damping(row):
    damping = 0.025 # as defined in prEN for CLT floors

    return damping

def calculate_frequency(row):
    k_e_1 = 1.0
    if span_type == 'one-way':
        k_e_2 = 1.0

    elif span_type == 'two-way':
        k_e_2 = math.sqrt(1 + ((((row['floor_span']) / (row['floor_width']))**4 * (row['D22'])) / (row['D11'])))

    mass = (row['gewicht'] + row['permanent_load'] + 0.1 * row['variable_load'])
    frequency = k_e_1 * k_e_2 * (math.pi / (2 * (row['floor_span'])**2)) * math.sqrt((row['D11'] * 1000 )/ mass)

    return frequency

def calculate_modal_mass(row):
    mass = (row['gewicht'] + row['permanent_load'] + 0.1 * row['variable_load'])

    if span_type == 'one-way':
        modal_mass = (mass * row['floor_span'] * row['floor_width']) / 2

    elif span_type == 'two-way':
        modal_mass = (mass * row['floor_span'] * row['floor_width']) / 4

    else:
        modal_mass = 0

    return modal_mass

# CLEANING DATABASE

database_full['damping'] = database_full.apply(define_damping, axis = 1)
database_full['natural_frequency'] = database_full.apply(calculate_frequency, axis = 1)
database_full['modal_mass'] = database_full.apply(calculate_modal_mass, axis = 1)

def calculate_bending_unity_check(row):
    load = (1.35 * (row['gewicht'] + row['permanent_load']) + 1.5 * row['variable_load']) * 0.0098 # [kN/m]
    moment = load * row['floor_span'] ** 2 / 8 # [kNm]
    section_modulus = 2 * (row['D11'] * 1000 / 12000) / row['dikte'] # m**3
    stress = moment / section_modulus # [kN/m]
    unity_check = stress / bending_strength

    return unity_check

database_full['unity_check_bending'] = database_full.apply(calculate_bending_unity_check, axis = 1)

def acting_mass(row):
    mass = row['gewicht'] + row['permanent_load'] + 0.1 * row['variable_load']
    return mass

database_full['acting_mass'] = database_full.apply(acting_mass, axis = 1)

bending_unity_check_limit = 1
filtered_database_full = database_full[database_full['unity_check_bending'] < bending_unity_check_limit]

#natural_frequency_min_limit = 4.5
#filtered_database_full = database_full[database_full['natural_frequency'] > natural_frequency_min_limit]

filtered_database_full = filtered_database_full[(filtered_database_full['natural_frequency'] >= 4.5) & (filtered_database_full['natural_frequency'] <= 20)]

print(filtered_database_full)

#natural_frequency_max_limit = 20
#filtered_database_full = database_full[database_full['natural_frequency'] < natural_frequency_max_limit]





