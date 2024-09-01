import pandas as pd
import numpy as np
import opensees.openseespy as ops
import matplotlib.pyplot as plt

from prEN_Chapter_9 import filtered_database_full_prEN_ch9 as data

# Define tolerance for filtering data
tolerance = 1e-9

# Filter data for one panel configuration
data_one_panel = data[np.isclose(data['floor_width'], 2.7, atol=tolerance)]

# Initialize lists for node coordinates and elements
node_coords_list1 = []
element_list1 = []

# Generate node coordinates and elements for one panel configuration
for index, row in data_one_panel.iterrows():
    floor_span = row['floor_span']
    floor_width = row['floor_width']

    node1 = (1, 0.0, 0.0)
    node2 = (2, floor_span, 0.0)
    node3 = (3, 0.0, floor_width)
    node4 = (4, floor_span, floor_width)

    node_coords_list1.append([node1, node2, node3, node4])
    element1 = (1, 1, 2, 4, 3, row['dikte'], 1)
    element_list1.append(element1)  # Append element tuple directly

node_coords1 = pd.DataFrame({'node_coords': node_coords_list1})
elements1 = pd.DataFrame({'elements': element_list1})

# Combine node coordinates and elements with the input data
one_panel_input_coords = pd.concat([data_one_panel.reset_index(drop=True), node_coords1], axis=1)
one_panel_input = pd.concat([one_panel_input_coords, elements1], axis=1)


# Function to calculate the mass
def calculation_mass(row):
    mass = (row['gewicht'] + row['permanent_load'] + 0.1 * row['variable_load'])
    return mass


# Function to compute the equivalent modulus of elasticity
def compute_equivalent_E(EI, thickness):
    E = (EI * 12) / (thickness ** 3) * 1000000
    return E


# Apply mass and equivalent modulus calculations to the data
one_panel_input['acting_mass'] = one_panel_input.apply(calculation_mass, axis=1)
one_panel_input['E_longitudinal'] = one_panel_input.apply(lambda row: compute_equivalent_E(row['D11'], row['dikte']),axis=1)
one_panel_input['E_transverse'] = one_panel_input.apply(lambda row: compute_equivalent_E(row['D22'], row['dikte']), axis=1)

# Select a subset of the data for testing
dummy_input = one_panel_input.iloc[:1]



# Function to create and analyze the model
def model_one_panel(node_coords, elements, E_longitudinal, E_transverse, thickness, mass_per_area, nu=0.2):
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    # Create nodes
    for node in node_coords:
        ops.node(node[0], node[1], node[2], 0.0)
    print("Nodes created.")
    print("Node coordinates:", node_coords)

    # Define line supports (simply supported slab along width)
    # I AM SURE THIS IS WHERE IS GOING WRONG, IT IS CURRENTLY NOT POSSIBLE TO DEFINE LINE SUPPORTS ONLY WITH THE 4 NODES COORDINATES
    floor_width = max(node[2] for node in node_coords)
    for node in node_coords:
        x, y = node[1], node[2]
        if np.isclose(x, 0.0) or np.isclose(x, 2.0):
            ops.fix(node[0], 1, 1)  # Fix u and v DOFs, allow rotation
        if np.isclose(y, floor_width):
            ops.fix(node[0], 0, 1)  # Fix v DOF, allow u and rotation

    # Define material properties
    Gxy = 0.5 * (E_longitudinal + E_transverse) / (1 + nu)
    Gyz = Gzx = Gxy
    Ez = 1e-6 * min(E_longitudinal, E_transverse)
    vyz = vzx = nu

    # Define the orthotropic material
    ops.nDMaterial('ElasticOrthotropic', 1, E_longitudinal, E_transverse, Ez, nu, vyz, vzx, Gxy, Gyz, Gzx,
                   mass_per_area)
    print("Material defined.")

    # Define the section properties for ShellMITC4 element
    ops.section('ElasticMembranePlateSection', 1, E_longitudinal, nu, thickness)
    print("Section defined.")

    # Debug elements
    print("Elements before creating:", elements)

    # Ensure elements is a list of tuples
    for ele in elements:
        if isinstance(ele, tuple) and len(ele) == 7:
            print(f"Creating element with data: {ele}")
            eleTag, *eleNodes, thick, eleType = ele
            ops.element('ShellMITC4', eleTag, *eleNodes, thick, eleType)  # Unpack the element tuple
        else:
            print(f"Invalid element data: {ele}")

    print("Elements created.")
    print("Elements:", elements)

    # Apply self-weight as load
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    g = 9.81
    load_factor = mass_per_area * g / 1000
    for node in node_coords:
        ops.load(node[0], 0.0, 0.0, -load_factor, 0.0, 0.0, 0.0)  # applying load in negative z-direction
    print("Loads applied.")
    print("Load factor:", load_factor)

    # Perform static analysis
    ops.constraints('Plain')
    ops.numberer('Plain')
    ops.system('BandGeneral')
    ops.algorithm('Linear')
    ops.integrator('LoadControl', 1.0)
    ops.analysis('Static')
    ops.analyze(1)
    print("Static analysis done.")

    # Check mass and stiffness matrices
    mass_matrix = ops.printA('Mass')
    stiffness_matrix = ops.printA('Stiffness')
    print("Mass Matrix:", mass_matrix)
    print("Stiffness Matrix:", stiffness_matrix)

    #Perform eigenvalue analysis to find natural frequencies
    numEigen = 5
    eigenvalues = ops.eigen(numEigen)
    if eigenvalues is None:
        print("Eigenvalue analysis failed.")
        return [], node_coords

    frequencies = [np.sqrt(eigenvalue) / (2 * np.pi) for eigenvalue in eigenvalues]
    print(f"Natural frequencies: {frequencies}")

    return frequencies, node_coords


#Function to plot mode shapes#
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


# Run the model and plot mode shapes for each row in the subset
for index, row in dummy_input.iterrows():
    print(f"\nModel {index + 1}")
    node_coords = row['node_coords']
    elements = row['elements']  # Directly use the elements list
    E_longitudinal = row['E_longitudinal']
    E_transverse = row['E_transverse']
    thickness = row['dikte']
    mass_per_area = row['acting_mass']

    frequencies, node_coords = model_one_panel(node_coords, elements, E_longitudinal, E_transverse, thickness, mass_per_area)
    if frequencies:
        # Assuming mode shapes are obtained from the analysis and stored in some variable, e.g., mode_shapes
        mode_shapes = ops.nodeEigenvector(node_coords[0][0], 1)  # Replace with actual method to get mode shapes
        plot_mode_shape(node_coords, mode_shapes, 0)  # Plot the first mode shape


node_coords = one_panel_input['node_coords'].iloc[0]
element_tags = one_panel_input['elements'].iloc[0]

# Plot nodes
for node in node_coords:
    plt.plot(node[1], node[2], 'bo')  # Plot node in blue circle

# Plot elements
for tag in element_tags:
    # Find the row where the tag is present in the 'elements' column
    element_row = one_panel_input[one_panel_input['elements'] == tag]

    # If element with the tag exists
    if not element_row.empty:
        # Extract the element tuple from the DataFrame
        element = element_row['elements'].iloc[0]

        # Extract node indices for the element
        node_indices = element[1:5]

        # Get node coordinates for the element
        element_nodes = [node_coords[idx - 1] for idx in node_indices]

        # Plot element as a line connecting its nodes
        x_coords = [node[1] for node in element_nodes]
        y_coords = [node[2] for node in element_nodes]
        plt.plot(x_coords, y_coords, 'k-')  # Plot element in black line

# Plot continuous supports along the sides of the slab
floor_width = node_coords[2][2]  # Assuming floor width is constant
support_y = [0, floor_width]  # Y-coordinates for the supports
support_x = [0, 3]  # X-coordinates for the supports
plt.plot([0, 0], support_y, 'r-')  # Left support line
plt.plot([3, 3], support_y, 'r-')  # Right support line

# Set plot labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Structural Model Visualization')

# Set aspect ratio to equal and show plot
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.show()