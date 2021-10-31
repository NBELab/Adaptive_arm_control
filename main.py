import pickle
from collections import defaultdict
from time import sleep

from model import Model
from arm import Simulation
import numpy as np

model = Model('./arm_models/NBEL/NBEL.xml')

init_angles = {0: -np.pi / 2, 1: 0, 2: np.pi / 2, 3: 0, 4: np.pi / 2, 5: 0}
target = [
    np.array([0.20, 0.10, -0.10]),
    np.array([0.20, 0.10, 0.10]),
    np.array([-0.20, 0.10, -0.10]),
    np.array([-0.20, 0.10, 0.10]),
    np.array([0.20, -0.30, -0.10]),
    np.array([0.20, -0.30, 0.10]),
    np.array([-0.20, -0.30, -0.10]),
    np.array([-0.20, -0.30, 0.10])
]
# angles = [{1: 180, 2: 226, 3: 135, 4: 180, 5: 182, 6: 180, 7: 135, 8: 180, 9: 240},
#           {1: 180, 2: 191, 3: 170, 4: 170, 5: 192, 6: 90, 7: 90, 8: 90, 9: 240},
#           {1: 180, 2: 113, 3: 248, 4: 250, 5: 112, 6: 88, 7: 90, 8: 90, 9: 240},
#           {1: 123, 2: 207, 3: 154, 4: 155, 5: 207, 6: 91, 7: 90, 8: 90, 9: 240},
#           {1: 180, 2: 216, 3: 145, 4: 190, 5: 172, 6: 180, 7: 135, 8: 180, 9: 255},
#           {1: 178, 2: 193, 3: 168, 4: 192, 5: 170, 6: 179, 7: 172, 8: 180, 9: 255},
#           {1: 133, 2: 250, 3: 111, 4: 117, 5: 244, 6: 273, 7: 204, 8: 89, 9: 255},
#           {1: 180, 2: 193, 3: 168, 4: 191, 5: 170, 6: 178, 7: 156, 8: 187, 9: 255},
#           {1: 180, 2: 226, 3: 135, 4: 180, 5: 182, 6: 180, 7: 135, 8: 180, 9: 240},
#           {1: 180, 2: 286, 3: 75, 4: 90, 5: 272, 6: 180, 7: 135, 8: 180, 9: 240}]
#
# simulation_ext = Simulation(model, init_angles)

#
# def add_angles(x, y):
#     return (x + y) % (2 * np.pi) - np.pi
#
#
# def wrap_angle(x):
#     if -np.pi <= x <= np.pi:
#         return x
#     elif x > np.pi:
#         return x - 2 * np.pi
#     elif x < -np.pi:
#         return x + 2 * np.pi
#     else:
#         raise Exception(x)

#
# # Offset angles for the physical arm in relative to the IK mpdel
# offset_relative_to_IK_Model = {1: 90, 2: 180, 3: 180, 4: 180,
#                                5: 180, 6: 0, 7: 180, 8: 0, 9: 0}
# offset_relative_to_IK_Model_openu = {1: 181, 2: 180, 3: 180, 4: 180,
#                                      5: 180, 6: 0, 7: 180, 8: 0, 9: 0}

#
# def robot_to_model_position(robot_position, openu=False):
#     if openu:
#         offset = offset_relative_to_IK_Model_openu
#     else:
#         offset = offset_relative_to_IK_Model
#     return [np.deg2rad(robot_position[1] - offset[1]),
#             -1 * np.deg2rad(robot_position[3] - offset[3]),
#             -1 * np.deg2rad((360 - robot_position[4]) - offset[4]),
#             np.deg2rad(robot_position[6] - offset[6]),
#             -1 * np.deg2rad(robot_position[7] - offset[7])]
#
#
# def model_to_robot_position(model_position, openu=False):
#     if openu:
#         offset = offset_relative_to_IK_Model_openu
#     else:
#         offset = offset_relative_to_IK_Model
#     f = [(np.rad2deg(model_position[0]) + offset[1]) % 360,
#          (np.rad2deg(-1 * model_position[1]) + offset[2]) % 360,
#          360 - ((np.rad2deg(-1 * model_position[2]) + offset[4]) % 360),
#          (np.rad2deg(model_position[3]) + offset[6]) % 360,
#          (np.rad2deg(-1 * model_position[4]) + offset[7]) % 360]
#     return {1: f[0], 2: 361 - f[1], 3: f[1], 4: f[2],
#             5: 361 - f[2], 6: f[3], 7: f[4], 8: 180, 9: 180}


# for a in angles:
#     pos = robot_to_model_position(a, openu=True)
#     # pos2 = [wrap_angle(p) for p in pos]
#     pos = {i: p for i, p in enumerate(pos)}
#     for i in range(500):
#         simulation_ext.send_target_angles(pos)
#         simulation_ext.simulation.step()
#         simulation_ext.viewer.render()
#
#     # sleep(5)
#     simpos = simulation_ext.get_ee_position_from_sim()
#     simpos = str(simpos).replace(' ', ',').replace('[', '').replace(']', '').replace(',,', ',')
#     print(simpos)


nneurons = [100, 1000, 1000]
seperate_dim = [True, False]
tau = [(0.01, 0.1, 0.5, 1)]
for nn in nneurons:
    print(nn, ' neurons')





    '''
    adaptation code
    '''
    # simulation_ext = Simulation(model, init_angles, return_to_null=False, target=target, n_neurons=nn,
    #                             external_force=1.5,
    #                             adapt=False)
    # output = simulation_ext.simulate(1500)
    # with open(f'test_out/{nn}_no_adapt.pkl', 'wb') as fp:
    #     pickle.dump(output, fp)
    #
    # print("With adapt")
    # simulation_ext = Simulation(model, init_angles, return_to_null=False, target=target, n_neurons=nn,
    #                             external_force=1.5,
    #                             adapt=True)
    # output = simulation_ext.simulate(1500)
    # with open(f'test_out/{nn}_adapt.pkl', 'wb') as fp:
    #     pickle.dump(output, fp)
    # Simulation.show_adapt_exp()
