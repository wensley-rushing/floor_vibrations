import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from data_simulation import filtered_database_full

filtered_database_full_prEN_ch9 = filtered_database_full.copy()

def prEN_acceleration(row):
    k_res = max((0.19 * (row['floor_width'] / row['floor_span']) * (row['D11'] / row['D22'])**0.25), 1.0)
    a_rms = (k_res * 0.4 * 50) / (math.sqrt(2) * 2 * row['damping'] * row['modal_mass'])
    R_a_rms = a_rms / 0.005

    return R_a_rms

def prEN_velocity(row):
    I_mod_mean = (42 * 2**1.43) / (row['natural_frequency']**1.3) #WALKING FREQUENCY DEFINED AS 2 Hz
    v_1_peak = 0.7 * (I_mod_mean) / (row['modal_mass'] + 70)
    k_imp = max((0.48 * (row['floor_width'] / row['floor_span']) * (row['D11'] / row['D22'])**0.25), 1.0)
    if 1.0 <= k_imp <= 1.7:
        neta = 1.35 - 0.4 * k_imp
    else:
        neta = 0.67
    v_tot_peak = k_imp * v_1_peak
    v_rms = v_tot_peak * (0.65 - 0.01 * row['natural_frequency']) * (1.22 - 11 * row['damping']) * neta
    R_v_rms = v_rms / 0.0001

    return R_v_rms

def prEN_stiffness(row):
    b_ef = min((0.95 * row['floor_span'] * (row['D22'] / row['D11'])**0.25), row['floor_width'])
    w_1kN = row['floor_span'] / (48 * row['D11'] * b_ef)

    return w_1kN



