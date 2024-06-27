import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np

data_one_way = pd.read_excel('data_one_way_i3.xlsx')
data_two_way = pd.read_excel('data_two_way_i4.xlsx')

# HISTOGRAMS MODAL MASS
# ONE-WAY DATA

if 'modal_masses_per' in data_one_way.columns and 'floor_width' in data_one_way.columns:
    # Parse 'modal_masses_per' as lists if they are string representations
    data_one_way['parsed_modal_masses'] = data_one_way['modal_masses_per'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    def get_first_element(lst):
        if isinstance(lst, list) and len(lst) > 0:
            return lst[0]
        return None

    data_one_way['first_modal_mass'] = data_one_way['parsed_modal_masses'].apply(get_first_element)

    data_one_way = data_one_way.dropna(subset=['first_modal_mass'])

    unique_floor_widths = data_one_way['floor_width'].unique()

    unique_floor_widths.sort()

    all_first_modal_masses = data_one_way['first_modal_mass'].dropna().tolist()
    x_min = min(all_first_modal_masses)
    x_max = max(all_first_modal_masses)

    max_y = 0  # Initialize max_y to zero

    num_unique = len(unique_floor_widths)

    for floor_width in unique_floor_widths:
        subset = data_one_way[data_one_way['floor_width'] == floor_width]
        first_modal_masses = subset['first_modal_mass'].dropna().tolist()

        counts, bins = np.histogram(first_modal_masses, bins=10)

        if counts.max() > max_y:
            max_y = counts.max()

    n_cols = 2  # Number of columns in the subplot grid
    n_rows = (num_unique // n_cols) + (num_unique % n_cols > 0)  # Calculate number of rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))  # Adjust size as necessary

    axes = axes.flatten()

    for idx, floor_width in enumerate(unique_floor_widths):
        subset = data_one_way[data_one_way['floor_width'] == floor_width]

        first_modal_masses = subset['first_modal_mass'].dropna().tolist()

        axes[idx].hist(first_modal_masses, bins=10, edgecolor='black')
        axes[idx].set_title(f'Floor Width: {floor_width:.1f}')
        axes[idx].set_xlabel('Modal mass first mode (%)')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_xlim(x_min, x_max)  # Set the same x-axis range for each plot
        axes[idx].set_ylim(0, max_y)  # Set the same y-axis range for each plot
        axes[idx].grid(True)

    for i in range(num_unique, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('Modal_mass_distribution_one_way')
    # plt.show()


else:
    print("'modal_masses_per' or 'floor_width' column not found in the DataFrame.")

# TWO-WAY DATA

if 'modal_masses_per' in data_two_way.columns and 'floor_width' in data_two_way.columns:
    # Parse 'modal_masses_per' as lists if they are string representations
    data_two_way['parsed_modal_masses'] = data_two_way['modal_masses_per'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    def get_first_element(lst):
        if isinstance(lst, list) and len(lst) > 0:
            return lst[0]
        return None

    data_two_way['first_modal_mass'] = data_two_way['parsed_modal_masses'].apply(get_first_element)

    data_two_way = data_two_way.dropna(subset=['first_modal_mass'])

    unique_floor_widths = data_two_way['floor_width'].unique()

    unique_floor_widths.sort()

    all_first_modal_masses = data_two_way['first_modal_mass'].dropna().tolist()
    x_min = min(all_first_modal_masses)
    x_max = max(all_first_modal_masses)

    max_y = 0  # Initialize max_y to zero

    num_unique = len(unique_floor_widths)

    for floor_width in unique_floor_widths:
        subset = data_two_way[data_two_way['floor_width'] == floor_width]
        first_modal_masses = subset['first_modal_mass'].dropna().tolist()

        counts, bins = np.histogram(first_modal_masses, bins=10)

        if counts.max() > max_y:
            max_y = counts.max()

    n_cols = 2  # Number of columns in the subplot grid
    n_rows = (num_unique // n_cols) + (num_unique % n_cols > 0)  # Calculate number of rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))  # Adjust size as necessary

    axes = axes.flatten()

    for idx, floor_width in enumerate(unique_floor_widths):
        subset = data_two_way[data_two_way['floor_width'] == floor_width]

        first_modal_masses = subset['first_modal_mass'].dropna().tolist()

        axes[idx].hist(first_modal_masses, bins=10, edgecolor='black')
        axes[idx].set_title(f'Floor Width: {floor_width:.1f}')
        axes[idx].set_xlabel('Modal mass first mode (%)')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_xlim(x_min, x_max)  # Set the same x-axis range for each plot
        axes[idx].set_ylim(0, max_y)  # Set the same y-axis range for each plot
        axes[idx].grid(True)

    for i in range(num_unique, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('Modal_mass_distribution_two_way')
    # plt.show()

else:
    print("'modal_masses_per' or 'floor_width' column not found in the DataFrame.")



# if 'modal_masses_per' in data_one_way.columns:
#
#     first_modal_masses = []
#
#     for i, row in data_one_way.iterrows():
#         modal_masses = row['modal_masses_per']
#
#         # Convert the entry to a list if it's a string representation of a list
#         if isinstance(modal_masses, str):
#             try:
#                 modal_masses = ast.literal_eval(modal_masses)
#             except (ValueError, SyntaxError):
#                 print(f"Row {i}: Unable to parse modal_masses_per as a list. Value: {modal_masses}")
#                 continue
#
#         # Check if it is a list after conversion
#         if isinstance(modal_masses, list) and len(modal_masses) > 0:
#             first_modal_masses.append(modal_masses[0])
#         else:
#             print(f"Row {i}: modal_masses_per is not a list or is empty. Value: {modal_masses}")
#
#     # Print all first items
#     print("First items extracted from each row's modal_masses_per:")
#     print(first_modal_masses)
#
#     if first_modal_masses:  # Only plot if there's data
#         plt.hist(first_modal_masses, bins=10, edgecolor='black')
#         plt.xlabel('Modal Mass First Mode (%)')
#         plt.ylabel('Count')
#         plt.title('Histogram of First Modal Masses')
#         plt.grid(True)
#         plt.show()
#     else:
#         print("No valid data to plot.")
#
# else:
#     print("'modal_masses_per' column not found in the DataFrame.")


# print(data_one_way['modal_masses_per'][0])


# plt.figure(figsize=(8, 6))
# plt.hist(first_mode_mass, bins = 10, color = 'blue', edgecolor = 'black')
# plt.title('Histogram of First Mode Modal Masses')
# plt.grid(True)
# plt.show()















# -----------------------------------------------------------------------------
# col_widths = 'floor_width'
# col_modal_mass = 'modal_masses_per'
#
# print("Sample data from 'modal_masses_per' column:")
# print(data_one_way[col_modal_mass].head())
#
# floor_widths = [2.7, 5.4, 8.1, 10.8]
#
# plt.figure(figsize = (12, 8))
#
# for i, value in enumerate(floor_widths, start = 1):
#
#     filtered_data_one_way = data_one_way[data_one_way[col_widths] == value]
#
#     modal_masses = []
#
#     for row in filtered_data_one_way[col_modal_mass]:
#         modal_masses.extend(row)
#
#     plt.subplot(2, 2, i)
#     plt.hist(modal_masses, bins = 30, color = 'blue', edgecolor = 'black')
#     plt.title(f'First mode modal mass for floor width {value}')
#     plt.grid(True)
#
#
# plt.tight_layout()
# plt.show()



