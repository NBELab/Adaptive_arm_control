import pickle
from collections import defaultdict
from time import sleep

from mpl_toolkits.mplot3d import Axes3D

from model import Model
from arm import Simulation
import numpy as np

model = Model('./arm_models/NBEL/NBEL.xml')

init_angles = {0: -np.pi / 2, 1: 0, 2: np.pi / 2, 3: 0, 4: np.pi / 2, 5: 0}
# target = [[0.004358202720404778, 0.24968126663106002, 0.5839833688019348],
#           [0.2133086524552455, 0.30231730697157366, 0.4238612165636782],
#           [0.21829904526839308, 0.595476920837128, 0.19244056953420313],
#           [0.29799319788035894, -0.05902493579330757, 0.41552920456478243],
#           [0.005174658043600888, 0.2964559607701945, 0.6095851270425379],
#           [0.025071336042124102, 0.4687363346289538, 0.602869835708902],
#           [0.22613913482978693, 0.08993043535163768, 0.3282380230416459],
#           [0.011320422985028824, 0.47936992528843325, 0.5371649284032374],
#           [0.004358202720404778, 0.24968126663106002, 0.5839833688019348],
#           [0.002855880644787611, 0.16361329256190973, 0.07169673053224164]]
#
# simulation_ext = Simulation(model, init_angles, return_to_null=False, target=target, n_neurons=None, tau=None,
#                             external_force=None,
#                             adapt=False)
# output = simulation_ext.simulate(1000)
# with open(f'test_out/recording/pos.pkl', 'wb') as fp:
#     pickle.dump(output, fp)

# nneurons = [5000]
# seperate_dim = [True, False]
# taus = [0.01, 0.1, 1]
# n_targets = 40
# targetsx = np.random.uniform(-.20, .20, size=n_targets)
# targetsy = np.random.uniform(-.30, .10, size=n_targets)
# targetsz = np.random.uniform(-.10, .10, size=n_targets)
# target = np.array([targetsx, targetsy, targetsz]).T
# for nn in nneurons:
#     for tau in taus:
#         print(nn, ' neurons', tau, 'tau')
#         simulation_ext = Simulation(model, init_angles, return_to_null=True, target=target, n_neurons=nn, tau=tau,
#                                     external_force=None,
#                                     adapt=False)
#         output = simulation_ext.simulate(1000)
#         with open(f'test_out/compare/{nn}_{tau}_3_integrators_1_dim.pkl', 'wb') as fp:
#             pickle.dump(output, fp)
Simulation.plot_comparison()

################### adaptation code ######################

# nn = 1000
# n_targets = 100
# targetsx = np.random.uniform(-.20, .20, size=n_targets)
# targetsy = np.random.uniform(-.30, .10, size=n_targets)
# targetsz = np.random.uniform(-.10, .10, size=n_targets)
# target = np.array([targetsx, targetsy, targetsz]).T
#
# import matplotlib.pyplot as plt
#
# Axes3D
# ax = plt.figure().add_subplot(111, projection='3d')
# ax.scatter(0, 0, c='black', label='origin')
# ax.scatter(*np.array(target).T, c='red',label='targets')
# plt.legend()
# plt.title('Targets around origin')
# plt.show()
#
#
# # target = [
# # np.array([-0.06648099, -0.29834369, -0.05161849]),
# # np.array([ 0.14804096, -0.09987448,  0.06971978]),
# # np.array([-0.06623986,  0.01225466 , 0.09050459]),
# # np.array([ 0.0287609 ,  0.05786149, -0.02437095])
# # ]
# simulation_ext = Simulation(model, init_angles, return_to_null=True, target=target, n_neurons=nn,
#                             external_force=1.5,
#                             adapt=False)
# output = simulation_ext.simulate(1000)
# # with open(f'test_out/adaptation_exp/{nn}_no_adapt.pkl', 'wb') as fp:
# with open(f'test_out/adaptation_exp/{nn}_hist_no_adapt.pkl', 'wb') as fp:
#     pickle.dump(output, fp)
#
# print("With adapt")
# simulation_ext = Simulation(model, init_angles, return_to_null=True, target=target, n_neurons=nn,
#                             external_force=1.5,
#                             adapt=True)
# output = simulation_ext.simulate(1000)
# # with open(f'test_out/adaptation_exp/{nn}_adapt.pkl', 'wb') as fp:
# with open(f'test_out/adaptation_exp/{nn}_hist_adapt.pkl', 'wb') as fp:
#     pickle.dump(output, fp)
# # Simulation.show_adapt_exp(nn)
# # Simulation.adapt_vis_same(nn)
# #
# Simulation.plot_adaptive_histogram()
