import openseespy.opensees as ops

import pandas as pd
import numpy as np
import matplotlib as plt

from prEN_Chapter_9 import filtered_database_full_prEN_ch9 as data

tolerance = 1e-9

data_two_panels = data[np.isclose(data['floor_width'], 5.4, atol=tolerance)]
data_three_panels = data[np.isclose(data['floor_width'], 8.1, atol=tolerance)]
data_four_panels = data[np.isclose(data['floor_width'], 10.8, atol=tolerance)]

#-----------------------------------------------------------------------------------
# THIS IS NOT THE MOST EFFICIENT WAY TO CLEAN THE DATA AND DEFINE THE MODEL, BUT MOST TIME EFFICIENT ATM
# 4 DIFFERENT MODELS WILL BE CREATED, ONE_PANEL, TWO_PANELS, THREE_PANELS, FOUR_PANELS
#-----------------------------------------------------------------------------------

# ONE PANEL
data_one_panel = data[np.isclose(data['floor_width'], 2.7, atol=tolerance)]
node_coords_list1 = []
element_list1 = []

for index, row in data_one_panel.iterrows():
    floor_span = row['floor_span']
    floor_width = row['floor_width']

    node1 = (1, 0.0, 0.0)
    node2 = (2, floor_span, 0.0)
    node3 = (3, 0.0, floor_width)
    node4 = (4, floor_span, floor_width)

    node_coords_list1.append([node1, node2, node3, node4])
    element_list1.append([1, 1, 2, 3, 4])

node_coords1 = pd.DataFrame({'node_coords': node_coords_list1})
elements1 = pd.DataFrame({'elements': element_list1})

one_panel_input_coords = pd.concat([data_one_panel.reset_index(drop=True), node_coords1], axis=1)
one_panel_input = pd.concat([one_panel_input_coords, elements1], axis=1)

def calculation_mass(row): #mass of the floor is taken as self + permanent + 10% variable in kg/m**2
    mass = (row['gewicht'] + row['permanent_load'] + 0.1 * row['variable_load'])

    return mass

one_panel_input['acting_mass'] = one_panel_input.apply(calculation_mass, axis = 1)


def model_one_panel(node_coords, elements, E_longitudinal, E_transverse, mass):
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    for node in node_coords:
        ops.node(node[0], node[1], node[2], 0.0)

    # define line supports (simply supported slab along width)
    floor_width = node_coords[2][2]
    for node in node_coords:
        if node[2] == 0.0 or node[2] == floor_width:
            ops.fix(node[0], 1, 1, 1, 0, 0, 0)

    # define material properties (CHECK THIS)
    nu = 0.2

    # define section properties # CHECK IF THIS IS CORRECTLY DEFINED
    ops.section('ElasticOrthotropic', 1, E1 = E_longitudinal, E2 = E_transverse, nu12 = nu, G12 = 0.0, rho = mass)

    # define elements
    for ele in elements:
        ops.element('ShellMITC4', ele[0], ele[1], ele[2], ele[3], ele[4], 1)



# start model generation


