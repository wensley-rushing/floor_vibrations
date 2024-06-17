import pandas as pd
import numpy as np
import openseespy.opensees as ops
import matplotlib.pyplot as plt
from itertools import count
from collections import defaultdict
from prEN_Chapter_9 import filtered_database_full_prEN_ch9 as data

def compute_equivalent_E(EI, thickness):
    E = (EI * 12) / (thickness ** 3) * 1000000 * 1000000
    return E


data['E_longitudinal'] = data.apply(lambda row: compute_equivalent_E(row['D11'], row['dikte']), axis=1)
data['E_transverse'] = data.apply(lambda row: compute_equivalent_E(row['D22'], row['dikte']), axis=1)

dummy_data = data.iloc[31:32].copy()

def model_two_way(floor_span, floor_width, thickness, mass_per_area, E_long, E_trans, mesh_size=0.9, shell=True, output=False):
    try:
        thickness = (thickness / 1000) # m
        div = (floor_width / 2.7) - 1

        xdicr = int(floor_span / mesh_size) + 1
        ydicr = int(floor_width / mesh_size) + 1

        x = np.linspace(0, floor_span, xdicr)
        y = np.linspace(0, floor_width, ydicr)

        xx, yy = np.meshgrid(x, y)

        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)

        nums = (np.arange(0, len(xx.flatten())) + 1).reshape(xx.shape)
        node_nums = [int(a) for a in nums.flatten()]
        all_nodes = nums.flatten()

        mid = int(all_nodes[int(len(all_nodes) / 2)])

        # if output:
        #     print('Mid plate point:')
        #     print(xx.flatten()[mid-1], yy.flatten()[mid-1])

        l_b = nums[:-1, :-1]
        r_b = nums[:-1, 1:]
        r_t = nums[1:, 1:]
        l_t = nums[1:, :-1]

        locs = [l_b, r_b, r_t, l_t]
        element_nodes = np.array([a.flatten() for a in locs]).T

        fixed = np.unique([b for a in [nums[0, :], nums[-1, :], nums[:, 0], nums[:, -1]] for b in a])

        nodes = {}
        for i, (x, y) in enumerate(zip(xx.flatten(), yy.flatten()), 1):
            nodes[i] = (x, y)
            ops.node(i, x, y, 0)
            if i in fixed:
                ops.fix(i, 1, 1, 1, 0, 0, 0)

        masses = defaultdict(float)

        if shell:
            e_type = 'ShellMITC4'
            mat_num = 1
            matTag = 1

            # Placeholder material properties
            Ex, Ey, Ez = E_long, E_trans, 430E6
            nu_xy, nu_yz, nu_zx = 0.372, 0.435, 0.02
            Gxy, Gyz, Gzx = 640E6, 30E6, 610E6
            rho = mass_per_area

            ops.nDMaterial('ElasticOrthotropic', matTag, Ex, Ey, Ez, nu_xy, nu_yz, nu_zx, Gxy, Gyz, Gzx, rho)
            ops.section('PlateFiber', mat_num, matTag, thickness)

            for e, _nodes in enumerate(element_nodes, 1):
                e_nodes = _nodes
                if div > 0 and nodes[_nodes[0]][1] % div < 0.00001 and nodes[_nodes[0]][1] > 0:
                    n1, n2 = int(_nodes[0]), int(_nodes[1])
                    offset = 100000

                    if n1 + offset not in nodes:
                        nodes[n1 + offset] = nodes[_nodes[0]]
                        ops.node(n1 + offset, *nodes[_nodes[0]], 0)
                        ops.equalDOF(n1, n1 + offset, 1, 2, 3)

                    if n2 + offset not in nodes:
                        nodes[n2 + offset] = nodes[_nodes[1]]
                        ops.node(n2 + offset, *nodes[_nodes[1]], 0)
                        ops.equalDOF(n2, n2 + offset, 1, 2, 3)

                    n1 += offset
                    n2 += offset
                    e_nodes = [n1, n2, _nodes[2], _nodes[3]]

                ops.element(e_type, e, *[int(a) for a in e_nodes], mat_num)
                area = mesh_size ** 2
                for a in e_nodes:
                    masses[int(a)] += area / 4 * mass_per_area

        else:
            eleTag = count()
            transfTag = 10
            ops.geomTransf('Linear', transfTag, 0.0, 0.0, 1.0)

            for a in nums:
                A = thickness * mesh_size
                E = 10000
                G = 640

                _b = min(mesh_size, thickness) / 2
                _a = max(mesh_size, thickness) / 2

                J = _a * _b**3 * (16/3 - 3.36 *_b/_a *(1-_b**4 /(12*_a**4)))
                J /= 10
                Iy = thickness**3 / 12 * mesh_size
                Iz = mesh_size**3 / 12 * thickness

                for n1, n2 in zip(a, a[1:]):
                    extra = []
                    if div > 0 and nodes[n2][1] % div < 0.00001 and nodes[n2][1] < floor_width:
                        extra = ['-releasey', 2]

                    ops.element('elasticBeamColumn', next(eleTag), int(n1), int(n2), A, E, G, J, Iy, Iz, transfTag, *extra)

        for _node, mass in masses.items():
            ops.mass(_node, *[mass] * 3, 0, 0, 0)

        numEigen = 5
        eigenValues = np.array(ops.eigen(numEigen))

        if eigenValues is None or len(eigenValues) == 0:
            print("Error: Eigenvalue computation failed or no values.")
            return None, None

        freqs = eigenValues**0.5 / (2 * np.pi)

        #if output:
        #    print(f"Computed Eigenvalues: {eigenValues}")
        #    print(f"Computed Frequencies: {freqs}")

        ops.record()

        mass_dist = np.array([masses[a] for a in node_nums])
        modal_masses_percentage = dict()

        # if output:
        #     print('\tMode\tFreq\tMass Percentage')
        #
        # fig, axes = plt.subplots(1, numEigen, figsize=(20, 4))
        # if numEigen == 1:
        #     axes = [axes]

        for i in range(numEigen):
            ev_data = np.array([ops.nodeEigenvector(a, i + 1, 3) for a in node_nums])
            ev_data /= np.max(np.abs(ev_data))
            zz = ev_data[nums - 1]

            modal_masses_percentage[i] = np.sum(ev_data ** 2 * mass_dist) / np.sum(mass_dist)

            if output:
                print(f'\t{i:5}\t{freqs[i]:5.2f}\t{modal_masses_percentage[i] * 100:5.1f}%')

        #     # Plot mode shape
        #     c = axes[i].contourf(xx, yy, zz)
        #     axes[i].set_title(f'Mode {i + 1}')
        #     axes[i].set_xlabel('X Position')
        #     axes[i].set_ylabel('Y Position')
        #     fig.colorbar(c, ax=axes[i], orientation='vertical', label='Displacement')
        #
        # plt.tight_layout()
        # plt.show()

        return freqs, modal_masses_percentage

    except ops.OpenSeesError as ose:
        print(f"OpenSees error: {ose}")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

# Iterate over the dummy data and apply the model
# for index, row in dummy_data.iterrows():
#    floor_span = row['floor_span']
#    floor_width = row['floor_width']
#    thickness = row['dikte']
#    mass_per_area = row['acting_mass']
#
#    frequencies, modal_masses = model_two_way(floor_span, floor_width, thickness, mass_per_area, output=True)
#    if frequencies is not None and modal_masses is not None:
#        for mode in range(len(frequencies)):
#            print(f"Mode {mode + 1}: Frequency = {frequencies[mode]:.2f} Hz, Modal Mass = {modal_masses[mode] * 100:.1f}%")

freq_lists = []
mass_lists = []

for index, row in dummy_data.iterrows():
    floor_span = row['floor_span']
    floor_width = row['floor_width']
    thickness = row['dikte']
    mass_per_area = row['acting_mass']
    E_long = row['E_longitudinal']
    E_trans = row['E_transverse']


    frequencies, modal_masses = model_two_way(floor_span, floor_width, thickness, mass_per_area, E_long, E_trans, output=True)
    if frequencies is not None and modal_masses is not None:
        freq_lists.append(frequencies.tolist())
        mass_lists.append(list(modal_masses.values()))

        for mode in range(len(frequencies)):
            print(f"Mode {mode + 1}: Frequency = {frequencies[mode]:.2f} Hz, Modal Mass = {modal_masses[mode] * 100:.1f}%")

# Append the lists to the DataFrame
dummy_data.loc[:, 'frequencies'] = freq_lists
dummy_data.loc[:, 'modal_masses'] = mass_lists

# print(dummy_data)

