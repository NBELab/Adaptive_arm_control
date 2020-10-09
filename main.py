from arm import Model, Simulation
import numpy as np

BASE_DIR = 'C:\\Users\\estis\\Adaptive_arm_control\\'
  
model_name  = 'NBEL'
model = Model(BASE_DIR + 'arm_models\{}\{}.xml'.format(model_name, model_name))

init_angles = {0: -np.pi/2, 1:0, 2:np.pi/2, 3:0, 4:np.pi/2, 5:0}
target      = [np.array([ 0.20 , 0.10,-0.10]), 
            #    np.array([ 0.20 , 0.10, 0.10]), ]
            #    np.array([-0.20 , 0.10,-0.10]), 
            #    np.array([-0.20 , 0.10, 0.10]),
            #    np.array([ 0.20 ,-0.30,-0.10]), 
            #    np.array([ 0.20 ,-0.30, 0.10]),  
            #    np.array([-0.20 ,-0.30,-0.10]), 
               np.array([-0.20 ,-0.30, 0.10])]


simulation_ext = Simulation(model, init_angles, external_force=1.5,
                            target=target, adapt=True)
simulation_ext.simulate(3500)
# simulation_ext.show_monitor()