import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib as plt


from prEN_Annex_G import df_full

filtered_database_full_copy = df_full.copy()


# extract CLT list from excel

ES_RMS_table_full = pd.read_excel('Vloertrilling_prEN_SBR_EC5_17102023.xlsx', sheet_name = 'ES_RMS_90_full')
ES_RMS_table = ES_RMS_table_full[ES_RMS_table_full['damping'] == 0.025].copy()
ES_RMS_table.set_index('f', inplace = True)
ES_RMS_table.drop('damping', axis = 1, inplace = True)


x = ES_RMS_table.columns.astype(float).values
y = ES_RMS_table.index.astype(float).values

interpolator = RegularGridInterpolator((y, x), ES_RMS_table.values, method='linear')

x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

def calculate_nat_freq_SBR(row):
    span_type = row['span_type']

    if span_type == 'one-way':
        nat_freq_SBR = (np.pi / (2 * row['floor_span'] ** 2)) * np.sqrt(row['D11'] * 1000 / row['acting_mass'])

    elif span_type == 'two-way':
        nat_freq_SBR = (np.pi / (2 * row['floor_span']**2)) * np.sqrt(row['D11'] * 1000 / row['acting_mass']) * np.sqrt(1 + (2 * (row['floor_span'] / row['floor_width'])**2 + (row['floor_span'] / row['floor_width'])**4) * (row['D22'] / row['D11']))

    return nat_freq_SBR

filtered_database_full_copy.loc[:, 'nat_freq_SBR'] = filtered_database_full_copy.apply(calculate_nat_freq_SBR, axis=1)
# Define a function to perform the interpolation
def interpolate_value(row):
    x_val = row['modal_mass_prEN_Ch9']
    y_val = row['nat_freq_SBR']
    # Clip values to be within the grid bounds
    x_val_clipped = np.clip(x_val, x_min, x_max)
    y_val_clipped = np.clip(y_val, y_min, y_max)
    return interpolator((y_val_clipped, x_val_clipped))

filtered_database_full_copy.loc[:, 'ES_RMS_value'] = filtered_database_full_copy.apply(interpolate_value, axis=1)

ES_RMS_limits = {
    'low_lim': [0.0, 0.1, 0.2, 0.8, 3.2, 12.8],
    'high_lim': [0.1, 0.2, 0.8, 3.2, 12.8, 51.2]
}

response_classes = ['A', 'B', 'C', 'D', 'E', 'F']

SBR_limits = pd.DataFrame(ES_RMS_limits, index = response_classes)

# print(SBR_limits)

response_class_list = []

for index, row in filtered_database_full_copy.iterrows():
    es_rms_value = row['ES_RMS_value']

    for cls, limits in SBR_limits.iterrows():
        low_lim = limits['low_lim']
        high_lim = limits['high_lim']

        if low_lim <= es_rms_value < high_lim:
            response_class = cls
            break

    response_class_list.append(response_class)

filtered_database_full_copy['response_class'] = response_class_list

# print(filtered_database_full_copy)

filtered_database_full_copy.to_excel('data_one_way_i4.xlsx', index = False)

#--------------------------------------------------------------------------------------------------------------
#
#
# colors = {'A': 'red', 'B': 'blue', 'C': 'green', 'D': 'orange', 'E': 'purple', 'F': 'black'}
#
# plt.figure(figsize=(6,10))
#
# for cls, color in colors.items():
#     subset = filtered_database_full_copy[filtered_database_full_copy['response_class'] == cls]
#     plt.scatter(subset['modal_mass'], subset['natural_frequency'], color = color, label = cls)
#
# plt.xscale('log')
# plt.yscale('log')
#
# plt.xlabel('Modal Mass')
# plt.ylabel('Natural Frequency')
# plt.title('Scatter Plot of Modal Mass vs Natural Frequency')
# plt.legend(title='Response Class')
# plt.show()

