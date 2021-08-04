import numpy as np
import matplotlib.pyplot as plt
import os
import nengo

from SNN.networks.integrator_array import IntegratorArray

N_INTEGRATORS = 1  # only ee for now
DIMENSIONS = 3


class EndEffectorModel:
    def __init__(self, n_neurons, tau, transform, inp_synapse=None, height0=np.zeros(DIMENSIONS)):
        self.model = nengo.Network()
        self.h_change = np.zeros(DIMENSIONS)
        self.heights0 = height0
        with self.model:
            self.stim = nengo.Node(lambda t: self.h_change)

            self.integrators = IntegratorArray(n_neurons=n_neurons,
                                               n_ensembles=N_INTEGRATORS,
                                               ens_dimensions=DIMENSIONS,  # xyz
                                               recurrent_tau=tau,
                                               inp_transform=transform,
                                               inp_synapse=inp_synapse,
                                               radius=1)

            nengo.Connection(self.stim, self.integrators.input, synapse=None)

            self.probe_out = nengo.Probe(self.integrators.output, synapse=0.01)
            self.dt = 0.002
            self.sim = nengo.Simulator(self.model, dt=self.dt)
            self.sim.step()

    def update(self, h_change):
        self.h_change = h_change #/ self.dt
        self.sim.step()

    def get_curr_pos(self):
        return (self.sim.data[self.probe_out] + self.heights0)[-1]

    def get_xy(self):
        return self.sim.trange(), (self.sim.data[self.probe_out] + self.heights0)

    @property
    def curr_val(self):
        return self.sim.data[self.probe_out][-1] + self.heights0

    def save_figs(self):
        path = os.path.dirname(__file__)
        figname = 'integrator_out.png'
        plt.figure()
        plt.title(figname)
        plt.grid()
        plt.plot(*self.get_xy())
        plt.savefig(os.path.join(path, 'out', figname))
