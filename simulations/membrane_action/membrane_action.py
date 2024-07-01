import openseespy.opensees as ops
import numpy as np

ops.wipe()

ops.model('basic', '-ndm', 3, '-ndf', 6)

node_coords = {
    1: (0, 0, 0),
    2: (5, 0, 0),
    3: (10, 0, 0),
    4: (0, 8.1, 0),
    5: (5, 8.1, 0),
    6: (10, 8.1, 0),
    7: (0, 16.2, 0),
    8: (5, 16.2, 0),
    9: (10, 16.2, 0),
    10: (0, 0, 3),
    11: (5, 0, 3),
    12: (10, 0, 3),
    13: (0, 8.1, 3),
    14: (5, 8.1, 3),
    15: (10, 8.1, 3),
    16: (0, 16.2, 3),
    17: (5, 16.2, 3),
    18: (10, 16.2, 3),
}

for node, coords in node_coords.items():
    ops.node(node, *coords)

# apply fixed supports at base nodes (1-9)

for i in range(1-10):
    ops.fix(i, 1, 1, 1, 1, 1, 1)