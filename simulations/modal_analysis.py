import pandas as pd
import numpy as np
import openseespy.opensees as ops
import matplotlib.pyplot as plt

from prEN_Chapter_9 import filtered_database_full_prEN_ch9 as data

# Define tolerance for filtering data
tolerance = 1e-9

# Filter data for different panel configurations
data_two_panels = data[np.isclose(data['floor_width'], 5.4, atol=tolerance)]
data_three_panels = data[np.isclose(data['floor_width'], 8.1, atol=tolerance)]
data_four_panels = data[np.isclose(data['floor_width'], 10.8, atol=tolerance)]
data_one_panel = data[np.isclose(data['floor_width'], 2.7, atol=tolerance)]

# Initialize lists for node coordinates and elements
node_coords_list1 = []
element_list1 = []

# Generate node coordinates and elements for one panel configuration
for index, row in data_one_panel.iterrows():
    floor_span = row['floor_span']
    floor_width = row['floor_width']

    node1 = (1 + index * 4, 0.0, 0.0)
    node2 = (2 + index * 4, floor_span, 0.0)
    node3 = (3 + index * 4, 0.0, floor_width)
    node4 = (4 + index * 4, floor_span, floor_width)

    node_coords_list1.append([node1, node2, node3, node4])
    element_list1.append([1 + index, 1 + index * 4, 2 + index * 4, 4 + index * 4, 3 + index * 4])

node_coords1 = pd.DataFrame({'node_coords': node_coords_list1})
elements1 = pd.DataFrame({'elements': element_list1})

# Combine node coordinates and elements with the input data
one_panel_input_coords = pd.concat([data_one_panel.reset_index(drop=True), node_coords1], axis=1)
one_panel_input = pd.concat([one_panel_input_coords, elements1], axis=1)

# Calculate the acting mass for each row in the data
def calculation_mass(row):
    mass = (row['gewicht'] + row['permanent_load'] + 0.1 * row['variable_load'])
    return mass

# Calculate equivalent Young's modulus from effective bending stiffness and thickness
def compute_equivalent_EI(EI, thickness):
    E = (EI * 12) / (thickness ** 3)
    return E

# Apply mass and equivalent modulus calculations to the data
one_panel_input['acting_mass'] = one_panel_input.apply(calculation_mass, axis=1)
one_panel_input['E_longitudinal'] = one_panel_input.apply(lambda row: compute_equivalent_EI(row['D11'], row['dikte']), axis=1)
one_panel_input['E_transverse'] = one_panel_input.apply(lambda row: compute_equivalent_EI(row['D22'], row['dikte']), axis=1)

# Define the function to create and analyze the model
def model_one_panel(node_coords, elements, E_longitudinal, E_transverse, thickness, mass_per_area, nu=0.2):
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    for node in node_coords:
        ops.node(node[0], node[1], node[2], 0.0)

    floor_width = node_coords[2][2]
    for node in node_coords:
        if node[2] == 0.0 or node[2] == floor_width:
            ops.fix(node[0], 1, 1, 1, 0, 0, 0)

    Gxy = 0.5 * (E_longitudinal + E_transverse) / (1 + nu)
    Gyz = Gzx = Gxy  # Assuming isotropic shear modulus for simplicity
    Ez = 1e-6 * min(E_longitudinal, E_transverse)  # A small value for Ez as it's 2D
    vyz = vzx = nu

    ops.nDMaterial('ElasticOrthotropic', 1, E_longitudinal, E_transverse, Ez, nu, vyz, vzx, Gxy, Gyz, Gzx, mass_per_area)
    ops.section('ElasticMembranePlateSection', 1, E_longitudinal, nu, thickness)

    for ele in elements:
        ops.element('ShellMITC4', ele[0], ele[1], ele[2], ele[3], ele[4], 1)

    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    g = 9.81
    load_factor = mass_per_area * g
    for node in node_coords:
        ops.load(node[0], 0.0, 0.0, -load_factor, 0.0, 0.0, 0.0)  # applying load in negative z-direction

    ops.constraints('Plain')
    ops.numberer('Plain')
    ops.system('BandGeneral')
    ops.algorithm('Linear')
    ops.integrator('LoadControl', 1.0)
    ops.analysis('Static')
    ops.analyze(1)

    numEigen = 5
    eigenvalues = ops.eigen(numEigen)
    frequencies = [np.sqrt(eigenvalue) / (2 * np.pi) for eigenvalue in eigenvalues]
    print(f"Natural frequencies: {frequencies}")

    return frequencies, node_coords

# Define function to plot mode shapes
def plot_mode_shape(node_coords, mode_shapes, mode_number):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = [node[1] for node in node_coords]
    y = [node[2] for node in node_coords]
    z = [mode_shapes[node[0] - 1][mode_number] for node in node_coords]

    ax.plot_trisurf(x, y, z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Mode Shape Amplitude')
    ax.set_title(f'Mode Shape {mode_number + 1}')
    plt.show()

# Select a subset of the data for testing
dummy_input = one_panel_input.iloc[:1]

# Run the model and plot mode shapes for each row in the subset
for index, row in dummy_input.iterrows():
    print(f"\nModel {index + 1}")
    node_coords = row['node_coords']
    elements = row['elements']

    frequencies, node_coords = model_one_panel(
        node_coords,
        [elements],
        row['E_longitudinal'],
        row['E_transverse'],
        row['dikte'],
        row['acting_mass']
    )

    numEigen = 5
    mode_shapes = [ops.nodeEigenvector(node[0], i + 1) for node in node_coords for i in range(numEigen)]
    mode_shapes = np.array(mode_shapes).reshape(len(node_coords), numEigen, 6)

    for mode_number in range(numEigen):
        plot_mode_shape(node_coords, mode_shapes[:, :, 2], mode_number)




