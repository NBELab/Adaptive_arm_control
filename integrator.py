import matplotlib.pyplot as plt
import nengo
import numpy as np
from nengo.processes import Piecewise
from mpl_toolkits.mplot3d import Axes3D

class Integrator:
    def __init__(self, **kwargs):
        self.nengo_model = nengo.Network()

        self.min_r = -1
        self.max_r = 1

        self.ee1_input = {0:0}
        self.ee2_input = {0:0}
        self.ee3_input = {0:0}
        self.ee4_input = {0:0}
        self.ee5_input = {0:0}

        def scale_down(x):
            return ((self.max_r-self.min_r) * (x - self.min_r) / (self.max_r - self.min_r)) + self.min_r
        def scale_up_deg(x):
            return ((self.max_r-self.min_r) * (x + 1) / 2) +  self.min_r
        
        def getInput(index):
            if index == 1:
                return self.ee1_input
            elif index == 2:
                return self.ee2_input
            elif index == 3:
                return self.ee3_input
            elif index == 4:
                return self.ee4_input
            elif index == 5:
                return self.ee5_input
            
        with self.nengo_model:
            ################################
            ######        ee1         ######
            ################################
            # representing an integrator of acceleration
            integrator_ee1 = nengo.Ensemble(n_neurons=1000, dimensions=1)
            # Create a piecewise step function for input
            stim_ee1  = nengo.Node(Piecewise(getInput(1)))

            # Connect the population to itself using a long time constant (tau) 
            # for stability
            tau = 0.1
            nengo.Connection(integrator_ee1, integrator_ee1, synapse=tau)

            # Connect the input using the same time constant as on the recurrent
            # connection to make it more ideal
            nengo.Connection(stim_ee1, integrator_ee1, transform=tau, synapse=tau)


            ################################
            ######         ee2        ######
            ################################
            # representing an integrator of acceleration
            integrator_ee2 = nengo.Ensemble(n_neurons=1000, dimensions=1)

            # Create a piecewise step function for input
            stim_ee2  = nengo.Node(Piecewise(getInput(2)))

            # Connect the population to itself using a long time constant (tau) 
            # for stability
            nengo.Connection(integrator_ee2, integrator_ee2, synapse=tau)

            # Connect the input using the same time constant as on the recurrent
            # connection to make it more ideal
            nengo.Connection(stim_ee2, integrator_ee2, transform=tau, synapse=tau)


            ################################
            ######         ee3        ######
            ################################
            # representing an integrator of acceleration
            integrator_ee3 = nengo.Ensemble(n_neurons=1000, dimensions=1)

            # Create a piecewise step function for input
            stim_ee3  = nengo.Node(Piecewise(getInput(3)))

            # Connect the population to itself using a long time constant (tau) 
            # for stability
            nengo.Connection(integrator_ee3, integrator_ee3, synapse=tau)

            # Connect the input using the same time constant as on the recurrent
            # connection to make it more ideal
            nengo.Connection(stim_ee3, integrator_ee3, transform=tau, synapse=tau)

            ################################
            ######         ee4        ######
            ################################
            # representing an integrator of acceleration
            integrator_ee4 = nengo.Ensemble(n_neurons=1000, dimensions=1)

            # Create a piecewise step function for input
            stim_ee4  = nengo.Node(Piecewise(getInput(4)))

            # Connect the population to itself using a long time constant (tau) 
            # for stability
            nengo.Connection(integrator_ee4, integrator_ee4, synapse=tau)

            # Connect the input using the same time constant as on the recurrent
            # connection to make it more ideal
            nengo.Connection(stim_ee4, integrator_ee4, transform=tau, synapse=tau)

            # ################################
            # ######         ee5        ######
            # ################################
            # representing an integrator of acceleration
            integrator_ee5 = nengo.Ensemble(n_neurons=1000, dimensions=1)

            # Create a piecewise step function for input
            stim_ee5  = nengo.Node(Piecewise(getInput(5)))

            # Connect the population to itself using a long time constant (tau) 
            # for stability
            nengo.Connection(integrator_ee5, integrator_ee5, synapse=tau)

            # Connect the input using the same time constant as on the recurrent
            # connection to make it more ideal
            nengo.Connection(stim_ee5, integrator_ee5, transform=tau, synapse=tau)

            # --------------------------------------------------------------------------
            # connect all the current coordinates to coherent ansamble to simplify view
            current_location = nengo.Ensemble(n_neurons=1000, dimensions=5)
            nengo.Connection(integrator_ee1, current_location[0], function=scale_up_deg)
            nengo.Connection(integrator_ee2, current_location[1], function=scale_up_deg)
            nengo.Connection(integrator_ee3, current_location[2], function=scale_up_deg)
            nengo.Connection(integrator_ee4, current_location[3], function=scale_up_deg)
            nengo.Connection(integrator_ee5, current_location[4], function=scale_up_deg)

            # connect probs
            stim_ee1_probe = nengo.Probe(stim_ee1)
            integrator_ee1_prob = nengo.Probe(integrator_ee1)
            self.integrator_ee1_current_location_prob = nengo.Probe(current_location[0])

            stim_ee2_probe = nengo.Probe(stim_ee2)
            integrator_ee2_prob = nengo.Probe(integrator_ee2)
            self.integrator_ee2_current_location_prob = nengo.Probe(current_location[1])

            stim_ee3_probe = nengo.Probe(stim_ee3)
            integrator_ee3_prob = nengo.Probe(integrator_ee3)
            self.integrator_ee3_current_location_prob = nengo.Probe(current_location[2])

            stim_ee4_probe = nengo.Probe(stim_ee4)
            integrator_ee4_prob = nengo.Probe(integrator_ee4)
            self.integrator_ee4_current_location_prob = nengo.Probe(current_location[3])

            stim_ee5_probe = nengo.Probe(stim_ee5)
            integrator_ee5_prob = nengo.Probe(integrator_ee5)
            self.integrator_ee5_current_location_prob = nengo.Probe(current_location[4])

        self.sim = nengo.Simulator(self.nengo_model, dt=0.001)

    def setInput(self, integrator_dict):
            # integrator_dict = np.load("integrator_dict.npy", allow_pickle=True).item()

            self.ee1_input = {x:integrator_dict[x][0] for x in integrator_dict.keys()}
            self.ee2_input = {x:integrator_dict[x][1] for x in integrator_dict.keys()}
            self.ee3_input = {x:integrator_dict[x][2] for x in integrator_dict.keys()}
            self.ee4_input = {x:integrator_dict[x][3] for x in integrator_dict.keys()}
            self.ee5_input = {x:integrator_dict[x][4] for x in integrator_dict.keys()}

    def generate(self, input_signal, time):

        self.sim_time = time

        self.setInput(input_signal)

        # run the simulation (applies to both setups)
        self.sim.run(self.sim_time)

        # plt.figure()
        # plt.title(label= 'current coordinates on each dimmention')
        # plt.plot(self.sim.trange(), self.sim.data[self.integrator_ee1_current_location_prob], label='ee1')
        # plt.plot(self.sim.trange(), self.sim.data[self.integrator_ee2_current_location_prob], label='ee2')
        # plt.plot(self.sim.trange(), self.sim.data[self.integrator_ee3_current_location_prob], label='ee3')
        # plt.plot(self.sim.trange(), self.sim.data[self.integrator_ee4_current_location_prob], label='ee4')
        # plt.plot(self.sim.trange(), self.sim.data[self.integrator_ee5_current_location_prob], label='ee5')
        # plt.legend(loc="best");
        # plt.show()

        EE1 = self.sim.data[self.integrator_ee1_current_location_prob].ravel()
        EE2 = self.sim.data[self.integrator_ee2_current_location_prob].ravel()
        EE3 = self.sim.data[self.integrator_ee3_current_location_prob].ravel()
        EE4 = self.sim.data[self.integrator_ee4_current_location_prob].ravel()
        EE5 = self.sim.data[self.integrator_ee5_current_location_prob].ravel()

        final_position = [EE1[-1], EE2[-1], EE3[-1], EE4[-1], EE5[-1]]
        # np.save("final_position.npy", final_position)

        # s = np.dstack([EE1, EE2, EE3, EE4, EE5])

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(EE1, EE2, EE3)
        # # ax.plot(X, Y, Z, label='3D current coordinates')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.show()  

        return final_position

    
# # Generate some test data
# data = s.reshape((5,2000,3))

# # Write the array to disk
# with open('current_coordinates.txt', 'w') as outfile:
#     # I'm writing a header here just for the sake of readability
#     # Any line starting with "#" will be ignored by numpy.loadtxt
#     outfile.write('# Array shape: {0}\n'.format(data.shape))
#     outfile.write('# Array cols: X, Y, Z\n')

#     # Iterating through a ndimensional array produces slices along
#     # the last axis. This is equivalent to data[i,:,:] in this case
#     for data_slice in data:

#         # The formatting string indicates that I'm writing out
#         # the values in left-justified columns 7 characters in width
#         # with 5 decimal places.  
#         np.savetxt(outfile, data_slice, fmt='%-7.5f')

#         # Writing out a break to indicate different slices...
#         outfile.write('# New slice\n')