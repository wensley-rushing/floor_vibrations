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

for index, row in data_one_panel.iterrows():
    floor_span = row['floor_span']
    floor_width = row['floor_width']

    node1 = (1, 0.0, 0.0)
    node2 = (2, floor_span, 0.0)
    node3 = (3, 0.0, floor_width)
    node4 = (4 + 4, floor_span, floor_width)

    node_coords_list1.append([node1, node2, node3, node4])

node_coords1 = pd.DataFrame({'node_coords': node_coords_list1})
one_panel_input = pd.concat([data_one_panel, node_coords1], axis = 1)

print(one_panel_input)

#def model_one_panel(node_coords, elements, E_longitudinal, E_transverse, rho):
#    ops.wipe()

#    ops.model('basic', '-ndm', 3, '-ndf', 6)



# start model generation


