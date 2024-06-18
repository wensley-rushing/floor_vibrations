import pandas as pd
import numpy as np
import math
import matplotlib as plt

from data_simulation import filtered_database_ULS

filtered_database_full_prEN_ch9 = filtered_database_ULS.copy()

# def calculate_frequency(row):
#
#     span_type = row['span_type']
#
#     k_e_1 = 1.0
#     if span_type == 'one-way':
#         k_e_2 = 1.0
#
#     elif span_type == 'two-way':
#         k_e_2 = math.sqrt(1 + ((((row['floor_span']) / (row['floor_width']))**4 * (row['D22'])) / (row['D11'])))
#
#     else:
#         k_e_2 = float('inf')
#
#     mass = row['acting_mass']
#     frequency = k_e_1 * k_e_2 * (math.pi / (2 * (row['floor_span'])**2)) * math.sqrt((row['D11'] * 1000 )/ mass)
#
#     return frequency
# #
# #
# def calculate_modal_mass(row):
#     mass = row['acting_mass']
#
#     span_type = row['span_type']
#
#     if span_type == 'one-way':
#         modal_mass = (mass * row['floor_span'] * row['floor_width']) / 2
#
#     elif span_type == 'two-way':
#         modal_mass = (mass * row['floor_span'] * row['floor_width']) / 4
#
#     else:
#         modal_mass = 0
#
#     return modal_mass
# #
# database_prEN['nat_freq_prEN_Ch9'] = database_prEN.apply(calculate_frequency, axis = 1)
# database_prEN['modal_mass_prEN_Ch9'] = database_prEN.apply(calculate_modal_mass, axis = 1)
#
# filtered_database_full_prEN_ch9 = database_prEN[(database_prEN['nat_freq_prEN_Ch9'] >= 4.5) & (database_prEN['nat_freq_prEN_Ch9'] <= 20)]
#
def prEN_acceleration(row):
    k_res = max((0.19 * (row['floor_width'] / row['floor_span']) * (row['D11'] / row['D22'])**0.25), 1.0)
    a_rms = (k_res * 0.4 * 50) / (math.sqrt(2) * 2 * row['damping'] * row['modal_mass_prEN_Ch9'])
    R_a_rms = a_rms / 0.005

    return R_a_rms

def prEN_velocity(row):
    I_mod_mean = (42 * 2**1.43) / (row['nat_freq_prEN_Ch9']**1.3) #WALKING FREQUENCY DEFINED AS 2 Hz
    v_1_peak = 0.7 * (I_mod_mean) / (row['modal_mass_prEN_Ch9'] + 70)
    k_imp = max((0.48 * (row['floor_width'] / row['floor_span']) * (row['D11'] / row['D22'])**0.25), 1.0)
    if 1.0 <= k_imp <= 1.7:
        neta = 1.35 - 0.4 * k_imp
    else:
        neta = 0.67
    v_tot_peak = k_imp * v_1_peak
    v_rms = v_tot_peak * (0.65 - 0.01 * row['nat_freq_prEN_Ch9']) * (1.22 - 11 * row['damping']) * neta
    R_v_rms = v_rms / 0.0001

    return R_v_rms

filtered_database_full_prEN_ch9['R_a_rms'] = filtered_database_full_prEN_ch9.apply(prEN_acceleration, axis = 1)
filtered_database_full_prEN_ch9['R_v_rms'] = filtered_database_full_prEN_ch9.apply(prEN_velocity, axis = 1)

def govenring_R(row):
    if row['nat_freq_prEN_Ch9'] < 8: #DEFINE WALKING FREQUENCY AS 2 Hz
        R_gov = max(row['R_v_rms'], row['R_a_rms'])

    else:
        R_gov = row['R_v_rms']

    return R_gov

filtered_database_full_prEN_ch9['R_gov'] = filtered_database_full_prEN_ch9.apply(govenring_R, axis = 1)

def prEN_stiffness(row):
    b_ef = min((0.95 * row['floor_span'] * (row['D22'] / row['D11'])**0.25), row['floor_width'])
    w_1kN = 1000 * (row['floor_span'] * 1000)**3 / (48 * row['D11']* 10**9 * b_ef) # output in mm

    return w_1kN

filtered_database_full_prEN_ch9['w_1kN'] = filtered_database_full_prEN_ch9.apply(prEN_stiffness, axis = 1)

comfort_limits = {
    'R_min_lim': [0.0, 4.0, 8.0, 12.0, 24.0, 36.0, 48.0],
    'R_max_lim': [4.0, 8.0, 12.0, 24.0, 36.0, 48.0, 1000.0],
    'w_lim_max': [0.25, 0.25, 0.5, 1.0, 1.5, 2.0, 2.0]
}

response_classes = ['I', 'II', 'III', 'IV', 'V', 'VI', 'X']

prEN_limits = pd.DataFrame(comfort_limits, index = response_classes)

prEN_comfort_class = []

for index, row in filtered_database_full_prEN_ch9.iterrows():
    R_value = row['R_gov']
    comfort_class = None
    for cls, limits in prEN_limits.iterrows():
        low_lim = limits['R_min_lim']
        high_lim = limits['R_max_lim']
        if low_lim <= R_value < high_lim:
            comfort_class = cls
            break
    prEN_comfort_class.append(comfort_class)

filtered_database_full_prEN_ch9['comfort_class'] = prEN_comfort_class

def check_stiffness_criteria(row, limits):
    comfort_class = row['comfort_class']
    if comfort_class in limits.index:
        limits = limits.loc[comfort_class]
        if row['w_1kN'] < limits['w_lim_max']:
            return True
        else:
            return False


filtered_database_full_prEN_ch9['within_limits'] = filtered_database_full_prEN_ch9.apply(lambda row: check_stiffness_criteria(row, prEN_limits), axis=1)


filtered_database_full_prEN_ch9 = filtered_database_full_prEN_ch9[filtered_database_full_prEN_ch9['within_limits']]


# print(filtered_database_full_prEN_ch9)












