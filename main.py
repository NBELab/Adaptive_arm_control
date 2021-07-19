from arm import Model, Simulation
import numpy as np


model = Model('./arm_models/NBEL/NBEL.xml')

init_angles = {0: -np.pi / 2, 1: 0, 2: np.pi / 2, 3: 0, 4: np.pi / 2, 5: 0}
target = [np.array([0.20, 0.10, -0.10]),
          #    np.array([ 0.20 , 0.10, 0.10]), ]
          #    np.array([-0.20 , 0.10,-0.10]),
          #    np.array([-0.20 , 0.10, 0.10]),
          #    np.array([ 0.20 ,-0.30,-0.10]),
          #    np.array([ 0.20 ,-0.30, 0.10]),
          #    np.array([-0.20 ,-0.30,-0.10]),
          np.array([-0.20, -0.30, 0.10])]

simulation_ext = Simulation(model, init_angles, external_force=1.5,
                            target=target, adapt=True)
simulation_ext.simulate(3500)
simulation_ext.show_monitor()