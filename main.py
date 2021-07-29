import pickle
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
nneurons = [10000]
for nn in nneurons:
    print(nn, ' neurons')
    simulation_ext = Simulation(model, init_angles, target=target, n_neurons=nn)
    output = simulation_ext.simulate(1500)
    # with open(f'test_out/{nn} output.pkl', 'wb') as fp:
    #     pickle.dump(output, fp)

    simulation_ext.show_monitor()
