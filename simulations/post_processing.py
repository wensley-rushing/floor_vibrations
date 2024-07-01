import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np


data_one_way = pd.read_excel('data_one_way_i4.xlsx')
data_two_way = pd.read_excel('data_two_way_i5.xlsx')
data_one_way_continuous = pd.read_excel('data_one_way_continuous.xlsx')

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

    # Drop rows where 'first_modal_mass' is NaN
    data_one_way = data_one_way.dropna(subset=['first_modal_mass'])

    # Determine unique floor widths
    unique_floor_widths = data_one_way['floor_width'].unique()
    unique_floor_widths.sort()

    # Calculate the global x-axis range for consistent bins
    all_first_modal_masses = data_one_way['first_modal_mass'].dropna().tolist()
    x_min = min(all_first_modal_masses)
    x_max = max(all_first_modal_masses)

    # Define the target bin width
    bin_width = 0.02  # Adjust this value as needed

    # Calculate the number of bins for the entire range based on the bin width
    num_bins = int((x_max - x_min) / bin_width)

    # Initialize max_y to zero for determining consistent y-axis limits
    max_y = 0

    # Compute the maximum y value across all histograms
    for floor_width in unique_floor_widths:
        subset = data_one_way[data_one_way['floor_width'] == floor_width]
        first_modal_masses = subset['first_modal_mass'].dropna().tolist()

        # Calculate histogram counts with the defined number of bins
        counts, _ = np.histogram(first_modal_masses, bins=num_bins, range=(x_min, x_max))

        if counts.max() > max_y:
            max_y = counts.max()

    # Determine the subplot grid size
    n_cols = 2  # Number of columns in the subplot grid
    n_rows = (len(unique_floor_widths) // n_cols) + (len(unique_floor_widths) % n_cols > 0)  # Rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))  # Adjust size as necessary
    axes = axes.flatten()

    for idx, floor_width in enumerate(unique_floor_widths):
        subset = data_one_way[data_one_way['floor_width'] == floor_width]
        first_modal_masses = subset['first_modal_mass'].dropna().tolist()

        # Plot histogram with consistent bins and axis limits
        axes[idx].hist(first_modal_masses, bins=num_bins, range=(x_min, x_max), edgecolor='black')
        axes[idx].set_title(f'Floor Width: {floor_width:.1f}')
        axes[idx].set_xlabel('Modal mass first mode (%)')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_xlim(x_min, x_max)  # Set the same x-axis range for each plot
        axes[idx].set_ylim(0, max_y)  # Set the same y-axis range for each plot
        axes[idx].grid(True)

    # Remove any empty subplots
    for i in range(len(unique_floor_widths), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('Modal_mass_distribution_one_way')  # Uncomment to save the figure
    # plt.show()  # Show the plot

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

    # Drop rows where 'first_modal_mass' is NaN
    data_two_way = data_two_way.dropna(subset=['first_modal_mass'])

    # Determine unique floor widths
    unique_floor_widths = data_two_way['floor_width'].unique()
    unique_floor_widths.sort()

    # Calculate the global x-axis range for consistent bins
    all_first_modal_masses = data_two_way['first_modal_mass'].dropna().tolist()
    x_min = min(all_first_modal_masses)
    x_max = max(all_first_modal_masses)

    # Define the target bin width
    bin_width = 0.02  # Adjust this value as needed

    # Calculate the number of bins for the entire range based on the bin width
    num_bins = int((x_max - x_min) / bin_width)

    # Initialize max_y to zero for determining consistent y-axis limits
    max_y = 0

    # Compute the maximum y value across all histograms
    for floor_width in unique_floor_widths:
        subset = data_two_way[data_two_way['floor_width'] == floor_width]
        first_modal_masses = subset['first_modal_mass'].dropna().tolist()

        # Calculate histogram counts with the defined number of bins
        counts, _ = np.histogram(first_modal_masses, bins=num_bins, range=(x_min, x_max))

        if counts.max() > max_y:
            max_y = counts.max()

    # Determine the subplot grid size
    n_cols = 2  # Number of columns in the subplot grid
    n_rows = (len(unique_floor_widths) // n_cols) + (len(unique_floor_widths) % n_cols > 0)  # Rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))  # Adjust size as necessary
    axes = axes.flatten()

    for idx, floor_width in enumerate(unique_floor_widths):
        subset = data_two_way[data_two_way['floor_width'] == floor_width]
        first_modal_masses = subset['first_modal_mass'].dropna().tolist()

        # Plot histogram with consistent bins and axis limits
        axes[idx].hist(first_modal_masses, bins=num_bins, range=(x_min, x_max), edgecolor='black')
        axes[idx].set_title(f'Floor Width: {floor_width:.1f}')
        axes[idx].set_xlabel('Modal mass first mode (%)')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_xlim(x_min, x_max)  # Set the same x-axis range for each plot
        axes[idx].set_ylim(0, max_y)  # Set the same y-axis range for each plot
        axes[idx].grid(True)

    # Remove any empty subplots
    for i in range(len(unique_floor_widths), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('Modal_mass_distribution_two_way')  # Uncomment to save the figure
    # plt.show()  # Show the plot

else:
    print("'modal_masses_per' or 'floor_width' column not found in the DataFrame.")

# ONE-WAY CONTINUOUS

if 'modal_masses_per' in data_one_way_continuous.columns and 'floor_width' in data_one_way_continuous.columns:
    # Parse 'modal_masses_per' as lists if they are string representations
    data_one_way_continuous['parsed_modal_masses'] = data_one_way_continuous['modal_masses_per'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    def get_first_element(lst):
        if isinstance(lst, list) and len(lst) > 0:
            return lst[0]
        return None

    data_one_way_continuous['first_modal_mass'] = data_one_way_continuous['parsed_modal_masses'].apply(get_first_element)

    # Drop rows where 'first_modal_mass' is NaN
    data_one_way_continuous = data_one_way_continuous.dropna(subset=['first_modal_mass'])

    # Determine unique floor widths
    unique_floor_widths = data_one_way_continuous['floor_width'].unique()
    unique_floor_widths.sort()

    # Calculate the global x-axis range for consistent bins
    all_first_modal_masses = data_one_way_continuous['first_modal_mass'].dropna().tolist()
    x_min = min(all_first_modal_masses)
    x_max = max(all_first_modal_masses)

    # Define the target bin width
    bin_width = 0.02  # Adjust this value as needed

    # Calculate the number of bins for the entire range based on the bin width
    num_bins = int((x_max - x_min) / bin_width)

    # Initialize max_y to zero for determining consistent y-axis limits
    max_y = 0

    # Compute the maximum y value across all histograms
    for floor_width in unique_floor_widths:
        subset = data_one_way_continuous[data_one_way_continuous['floor_width'] == floor_width]
        first_modal_masses = subset['first_modal_mass'].dropna().tolist()

        # Calculate histogram counts with the defined number of bins
        counts, _ = np.histogram(first_modal_masses, bins=num_bins, range=(x_min, x_max))

        if counts.max() > max_y:
            max_y = counts.max()

    # Determine the subplot grid size
    n_cols = 2  # Number of columns in the subplot grid
    n_rows = (len(unique_floor_widths) // n_cols) + (len(unique_floor_widths) % n_cols > 0)  # Rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))  # Adjust size as necessary
    axes = axes.flatten()

    for idx, floor_width in enumerate(unique_floor_widths):
        subset = data_one_way_continuous[data_one_way_continuous['floor_width'] == floor_width]
        first_modal_masses = subset['first_modal_mass'].dropna().tolist()

        # Plot histogram with consistent bins and axis limits
        axes[idx].hist(first_modal_masses, bins=num_bins, range=(x_min, x_max), edgecolor='black')
        axes[idx].set_title(f'Floor Width: {floor_width:.1f}')
        axes[idx].set_xlabel('Modal mass first mode (%)')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_xlim(x_min, x_max)  # Set the same x-axis range for each plot
        axes[idx].set_ylim(0, max_y)  # Set the same y-axis range for each plot
        axes[idx].grid(True)

    # Remove any empty subplots
    for i in range(len(unique_floor_widths), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('Modal_mass_distribution_one_way_continuous')  # Uncomment to save the figure
    # plt.show()  # Show the plot

else:
    print("'modal_masses_per' or 'floor_width' column not found in the DataFrame.")

# WIDTH-DEPTH RATIO

## ONE-WAY

if 'modal_masses_per' in data_one_way.columns and 'floor_width' in data_one_way.columns and 'floor_span' in data_one_way.columns:
    # Parse 'modal_masses_per' as lists if they are string representations
    data_one_way['parsed_modal_masses'] = data_one_way['modal_masses_per'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    def get_first_element(lst):
        if isinstance(lst, list) and len(lst) > 0:
            return lst[0]
        return None

    data_one_way['first_modal_mass'] = data_one_way['parsed_modal_masses'].apply(get_first_element)

    # Drop rows where 'first_modal_mass' is NaN
    data_one_way = data_one_way.dropna(subset=['first_modal_mass'])

    # Calculate the width-to-depth ratio
    data_one_way['width_to_depth_ratio'] = data_one_way['floor_width'] / data_one_way['floor_span']

    # Scatter plot of width-to-depth ratio against first_modal_mass
    plt.figure(figsize=(10, 6))
    plt.scatter(data_one_way['width_to_depth_ratio'], data_one_way['first_modal_mass'], c='green', alpha=0.5)
    plt.title('Width-to-Depth Ratio vs. First Modal Mass')
    plt.xlabel('Width-to-Depth Ratio')
    plt.ylabel('First Modal Mass (%)')
    plt.grid(True)

    # Fit a linear trend line
    z = np.polyfit(data_one_way['width_to_depth_ratio'], data_one_way['first_modal_mass'], 1)
    p = np.poly1d(z)
    plt.plot(data_one_way['width_to_depth_ratio'], p(data_one_way['width_to_depth_ratio']), "r--")

    # Calculate R-squared
    yhat = p(data_one_way['width_to_depth_ratio'])  # Predicted values
    y = data_one_way['first_modal_mass']
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Annotate the plot with R-squared value
    textstr = f'$R^2 = {r_squared:.3f}$'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig('width_to_depth_vs_first_modal_mass_one_way')  # Uncomment to save the figure
    # plt.show()  # Display the plot

else:
    print("Required columns ('modal_masses_per', 'floor_width', 'floor_span') not found in the DataFrame.")

## ONE-WAY CONTINUOUS

if 'modal_masses_per' in data_one_way_continuous.columns and 'floor_width' in data_one_way_continuous.columns and 'floor_span' in data_one_way_continuous.columns:
    # Parse 'modal_masses_per' as lists if they are string representations
    data_one_way_continuous['parsed_modal_masses'] = data_one_way_continuous['modal_masses_per'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    def get_first_element(lst):
        if isinstance(lst, list) and len(lst) > 0:
            return lst[0]
        return None

    data_one_way_continuous['first_modal_mass'] = data_one_way_continuous['parsed_modal_masses'].apply(get_first_element)

    # Drop rows where 'first_modal_mass' is NaN
    data_one_way_continuous = data_one_way_continuous.dropna(subset=['first_modal_mass'])

    # Calculate the width-to-depth ratio
    data_one_way_continuous['width_to_depth_ratio'] = data_one_way_continuous['floor_width'] / data_one_way_continuous['floor_span']

    # Scatter plot of width-to-depth ratio against first_modal_mass
    plt.figure(figsize=(10, 6))
    plt.scatter(data_one_way_continuous['width_to_depth_ratio'], data_one_way_continuous['first_modal_mass'], c='green', alpha=0.5)
    plt.title('Width-to-Depth Ratio vs. First Modal Mass')
    plt.xlabel('Width-to-Depth Ratio')
    plt.ylabel('First Modal Mass (%)')
    plt.grid(True)

    # Fit a linear trend line
    z = np.polyfit(data_one_way_continuous['width_to_depth_ratio'], data_one_way_continuous['first_modal_mass'], 1)
    p = np.poly1d(z)
    plt.plot(data_one_way_continuous['width_to_depth_ratio'], p(data_one_way_continuous['width_to_depth_ratio']), "r--")

    # Calculate R-squared
    yhat = p(data_one_way_continuous['width_to_depth_ratio'])  # Predicted values
    y = data_one_way_continuous['first_modal_mass']
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Annotate the plot with R-squared value
    textstr = f'$R^2 = {r_squared:.3f}$'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig('width_to_depth_vs_first_modal_mass_two_way')  # Uncomment to save the figure
    # plt.show()  # Display the plot

else:
    print("Required columns ('modal_masses_per', 'floor_width', 'floor_span') not found in the DataFrame.")


## TWO-WAY

if 'modal_masses_per' in data_two_way.columns and 'floor_width' in data_two_way.columns and 'floor_span' in data_two_way.columns:
    # Parse 'modal_masses_per' as lists if they are string representations
    data_two_way['parsed_modal_masses'] = data_two_way['modal_masses_per'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    def get_first_element(lst):
        if isinstance(lst, list) and len(lst) > 0:
            return lst[0]
        return None

    data_two_way['first_modal_mass'] = data_two_way['parsed_modal_masses'].apply(get_first_element)

    # Drop rows where 'first_modal_mass' is NaN
    data_two_way = data_two_way.dropna(subset=['first_modal_mass'])

    # Calculate the width-to-depth ratio
    data_two_way['width_to_depth_ratio'] = data_two_way['floor_width'] / data_two_way['floor_span']

    # Scatter plot of width-to-depth ratio against first_modal_mass
    plt.figure(figsize=(10, 6))
    plt.scatter(data_two_way['width_to_depth_ratio'], data_two_way['first_modal_mass'], c='green', alpha=0.5)
    plt.title('Width-to-Depth Ratio vs. First Modal Mass')
    plt.xlabel('Width-to-Depth Ratio')
    plt.ylabel('First Modal Mass (%)')
    plt.grid(True)

    # Fit a linear trend line
    z = np.polyfit(data_two_way['width_to_depth_ratio'], data_two_way['first_modal_mass'], 1)
    p = np.poly1d(z)
    plt.plot(data_two_way['width_to_depth_ratio'], p(data_two_way['width_to_depth_ratio']), "r--")

    # Calculate R-squared
    yhat = p(data_two_way['width_to_depth_ratio'])  # Predicted values
    y = data_two_way['first_modal_mass']
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Annotate the plot with R-squared value
    textstr = f'$R^2 = {r_squared:.3f}$'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig('width_to_depth_vs_first_modal_mass_two_way')  # Uncomment to save the figure
    plt.show()  # Display the plot

else:
    print("Required columns ('modal_masses_per', 'floor_width', 'floor_span') not found in the DataFrame.")




# FULL COLOR PLOTS

## ONE-WAY
colors_SBR = {'A': 'purple', 'B': 'blue', 'C': 'green', 'D': 'yellow', 'E': 'orange', 'F': 'red'}

plt.figure(figsize=(6,10))

for cls, color in colors_SBR.items():
    subset = data_one_way[data_one_way['response_class'] == cls]
    plt.scatter(subset['modal_mass_prEN_Ch9'], subset['nat_freq_SBR'], color = color, label = cls)

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Modal Mass')
plt.ylabel('Natural Frequency')
plt.title('SBR one-way')
plt.legend(title='Response Class')
plt.savefig('SBR_scatter_plot_one_way')


colors_prEN = {'I': 'purple', 'II': 'blue', 'III': 'green', 'IV': 'yellow', 'V': 'orange', 'VI': 'red', 'X': 'red'}

plt.figure(figsize=(6,10))

for cls, color in colors_prEN.items():
    subset = data_one_way[data_one_way['comfort_class'] == cls]
    plt.scatter(subset['modal_mass_prEN_Ch9'], subset['nat_freq_SBR'], color = color, label = cls)

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Modal Mass')
plt.ylabel('Natural Frequency')
plt.title('prEN Chapter 9 - one-way')
plt.legend(title='Response Class')
plt.savefig('prEN_Ch9_scatter_plot_one_way')


## TWO-WAY

plt.figure(figsize=(6,10))

for cls, color in colors_SBR.items():
    subset = data_two_way[data_two_way['response_class'] == cls]
    plt.scatter(subset['modal_mass_prEN_Ch9'], subset['nat_freq_SBR'], color = color, label = cls)

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Modal Mass')
plt.ylabel('Natural Frequency')
plt.title('SBR one-way')
plt.legend(title='Response Class')
plt.savefig('SBR_scatter_plot_one_way')


plt.figure(figsize=(6,10))

for cls, color in colors_prEN.items():
    subset = data_two_way[data_two_way['comfort_class'] == cls]
    plt.scatter(subset['modal_mass_prEN_Ch9'], subset['nat_freq_SBR'], color = color, label = cls)

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Modal Mass')
plt.ylabel('Natural Frequency')
plt.title('prEN Chapter 9 - two-way')
plt.legend(title='Response Class')
plt.savefig('prEN_Ch9_scatter_plot_one_way')


# CORRELATION PLOT ONE WAY
plt.figure(figsize=(10,6))

x = data_one_way['R_gov']
y = data_one_way['R_max_Annex_G']

plt.scatter(x, y, color = 'green', marker = '.')

coefficients = np.polyfit(x, y, 1)  # Linear fit (degree 1)
poly_eqn = np.poly1d(coefficients)
y_pred = poly_eqn(x)

# Plot the regression line
plt.plot(x, y_pred, color='red', label=f'Regression Line: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}')

# Calculate R-squared
y_mean = np.mean(y)
ss_tot = np.sum((y - y_mean) ** 2)  # Total sum of squares
ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
r_squared = 1 - (ss_res / ss_tot)

# Annotate the plot with R-squared
plt.text(min(x), max(y) * 0.9, f'$R^2 = {r_squared:.2f}$', fontsize=12, color='red')



plt.title('Scatter Plot with Regression Line and $R^2$')
plt.xlabel('Chapter 9')
plt.ylabel('Annex G')
plt.legend()
plt.grid(True)



# CORRELATION PLOT TWO WAY
plt.figure(figsize=(10,6))

x = data_two_way['R_gov']
y = data_two_way['R_max_Annex_G']

plt.scatter(x, y, color = 'green', marker = '.')

coefficients = np.polyfit(x, y, 1)  # Linear fit (degree 1)
poly_eqn = np.poly1d(coefficients)
y_pred = poly_eqn(x)

# Plot the regression line
plt.plot(x, y_pred, color='red', label=f'Regression Line: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}')

# Calculate R-squared
y_mean = np.mean(y)
ss_tot = np.sum((y - y_mean) ** 2)  # Total sum of squares
ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
r_squared = 1 - (ss_res / ss_tot)

# Annotate the plot with R-squared
plt.text(min(x), max(y) * 0.9, f'$R^2 = {r_squared:.2f}$', fontsize=12, color='red')



plt.title('Scatter Plot with Regression Line and $R^2$')
plt.xlabel('Chapter 9')
plt.ylabel('Annex G')
plt.legend()
plt.grid(True)

plt.show()


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



