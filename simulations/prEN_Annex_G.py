# IMPLEMENTATION OF THE ANNEX ON 'ALTERNATIVE METHOD FOR VIBRATION ANALYSIS OF FLOORS' BASED ON DOCUMENT DATE: 2023-09-26

# G.2   Scope and field of application
# (1) The detailed method applies to all floors including floors of irregular floor shapes
# NOTE 1    The method is normally applied using numerical dynamic analysis methods
# NOTE 2    The method performs well for cases where the mass of the walker is less than one tenth of the modal mass

# G.3 General
# (1) Floors with modes lower than up to 4 times the walking frequency should be checked for resonant response as well as for transient response
# (2) When checking floors for resonance the calculation procedure should be carried out for all possible walking frequencies
# (3) Floors with modes higher than 4 times the walking frequency should be checked for transient response only
# NOTE      Method can be applied assuming a single combination of walker and receiver locations (conservative)
# (4) When checking floors for transient response, the highest walking frequency should be used
#       - 1,5 Hz where walker cannot walk a distance of more than 5m unobstructed
#       - 2,0 Hz where walker can walk a distance of between 5m and 10m unobstructed
#       - 2,5 Hz where walker can walk a distance of 10m unobstructed
#  (5) Timber to timber connections may normally be modelled as pinned connections

# G.4 Transient response
# (1) All modes with frequencies up to twice the floor fundamental frequency or 25 Hz (whichever is lower) should be calculated, to obtain the modal mass, stiffness and frequency

def calculated_v_rms(df, column_name_1, column_name_2):
    if column_name_1 not in df.columns or column_name_2 not in df.columns:
        raise ValueError(f"column '{column_name}' does not exist in the Dataframe")

    results = []

    for index, row in df.iterrows():
        list_in_row_1 = row[column_name_1]
        list_in_row_2 = row[column_name_2]

        if isinstance(list_in_row_1, list) and isinstance(list_in_row_2, list):
            threshold = min (list_in_row_1[0], 25) # refer to G.4 (1)

            filtered_indices = [i for i, element in enumerate(list_in_row_1) if element >= threshold]
            filtered_list_1 = [list_in_row_1[i] for i in filtered_indices]
            filtered_list_2 = [list_in_row_2[i] for i in filtered_indices]

            if not filtered_list_1:
                df.at[index, column_name_1] = None
                continue

            walking_frequency = 2 # refer to G.3 (4)

            I_mod_ef = (54 * walking_frequency**(1.43)) / filtered_list_1[i]**(1.3)

            v_m_peak = I_mod_ef[i] / filtered_list_2[i]








