"""
Written by Dr. Elishai Ezra Tsur 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ 25.8.2020

This work is based on the implementation of nonlinear dynamics adaptation using Nengo,
as described in:
DeWolf, Travis, Terrence C. Stewart, Jean-Jacques Slotine, and Chris Eliasmith. 
"A spiking neural model of adaptive arm control." 
Proceedings of the Royal Society B: Biological Sciences 283, no. 1843 (2016): 20162134.

"""
import numpy as np
from scipy.special import beta, betainc, betaincinv
from scipy.linalg import svd

import nengo
from nengo.dists import Distribution, UniformHypersphere

class DynamicsAdaptation:
    """ Learns to account for  nmodelled forces, given training signals """

    def __init__(self,
        n_input,                    # Number of inputs (angles, velocities)
        n_output,                   # Number of outputs (forces for joints)
        n_neurons         = 1000,   # Number of neurons / ensemble
        n_ensembles       = 1,      # Number of ensembles 
        seed              = None,   # Seed for random number generation
        pes_learning_rate = 1e-6,   # Adaptation learning rate
        means             = None,   # Means and variances to scale data from -1 to 1 
        variances         = None,   # Outliers will be scaled outside the -1 to 1
        **kwargs
    ):

        self.n_neurons         = n_neurons
        self.n_ensembles       = n_ensembles
        self.pes_learning_rate = pes_learning_rate
        self.input_signal      = np.zeros(n_input)
        self.training_signal   = np.zeros(n_output)
        self.output            = np.zeros(n_output)
        np.random.seed         = seed
       
        # Accounting for spherical hyperspace
        n_input += 1 

        # Accounting for unknow means and variances ---------------------------------
        if means is not None and variances is None:
            variances = np.ones(means.shape)
        elif means is None and variances is not None:
            means = np.zeros(variances.shape)
        self.means = np.asarray(means)
        self.variances = np.asarray(variances)

        # Setting synapse time constants --------------------------------------------
        self.tau_input = 0.012     # Time constant for input connection
        self.tau_training = 0.012  # Time constant for training signal
        self.tau_output = 0.2      # Time constant for the output

        # Setting intercepts for the ensembles --------------------------------------
    
        # Generates intercepts for a d-dimensional ensemble, such that, given a
        # random uniform input (from the interior of the d-dimensional ball), the
        # probability of a neuron firing has the probability density function given
        # by rng.triangular(left, mode, right, size=n)
        triangular = np.random.triangular(
            left=0.35, mode=0.45, right=0.55, size=n_neurons * n_ensembles
        )
        intercepts = nengo.dists.CosineSimilarity(n_input + 2).ppf(1 - triangular)
        intercepts = intercepts.reshape((n_ensembles, n_neurons))
            
        # Setting weights for the ensembles -----------------------------------------
        # TODO: using presaved weights
        weights = np.zeros((self.n_ensembles, n_output, self.n_neurons))

        # Setting encoders for the ensembles ----------------------------------------
  
        # if NengoLib is installed, use it to optimize encoder placement
        try:
            encoders_dist = ScatteredHypersphere(surface=True)
        except ImportError:
            encoders_dist = nengo.Default
            print(
                "NengoLib not installed, encoder placement will "
                + "be sub-optimal."
            )
        encoders = encoders_dist.sample(n_neurons * n_ensembles, n_input)
        encoders = encoders.reshape(n_ensembles, n_neurons, n_input)

        # Defining the Nengo adaptive model ------------------------------------------
        
        self.nengo_model = nengo.Network(seed=seed)
        self.nengo_model.config[nengo.Ensemble].neuron_type = nengo.LIF()
        
        with self.nengo_model:

            def input_signals_func(t):
                return self.input_signal

            input_signals = nengo.Node(input_signals_func, size_out=n_input)

            def training_signals_func(t):
                return -self.training_signal

            training_signals = nengo.Node(training_signals_func, size_out=n_output)

            def output_func(t, x):
                self.output = np.copy(x)

            output = nengo.Node(output_func, size_in=n_output, size_out=0)

            self.adapt_ens = []
            self.conn_learn = []
            for ii in range(self.n_ensembles):
                self.adapt_ens.append(
                    nengo.Ensemble(
                        n_neurons=self.n_neurons,
                        dimensions=n_input,
                        intercepts=intercepts[ii],
                        radius=np.sqrt(n_input),
                        encoders=encoders[ii],
                        **kwargs,
                    )
                )

                # hook up input signal to adaptive population to provide context
                nengo.Connection(
                    input_signals, self.adapt_ens[ii], synapse=self.tau_input,
                )

                self.conn_learn.append(
                    nengo.Connection(
                        self.adapt_ens[ii].neurons,
                        output,
                        learning_rule_type=nengo.PES(pes_learning_rate),
                        transform=weights[ii],
                        synapse=self.tau_output,
                    )
                )

                # hook up the training signal to the learning rule
                nengo.Connection(
                    training_signals,
                    self.conn_learn[ii].learning_rule,
                    synapse=self.tau_training,
                )

        nengo.rc.set("decoder_cache", "enabled", "False")
        self.sim = nengo.Simulator(self.nengo_model, dt=0.001)

    def generate(self, input_signal, training_signal):
        """ Generates the control signal given joints' position, velocity and learning signal """

        # Accounting for unknown means
        if self.means is not None:
            input_signal = self.scale_inputs(input_signal)

        # Store local copies to feed in to the adaptive population
        self.input_signal = input_signal
        self.training_signal = training_signal

        # Run the simulation
        self.sim.run(time_in_seconds=0.001, progress_bar=False)

        return self.output

    def scale_inputs(self, input_signal):
        """ Scaling inputs using expected means and variances """
        
        scaled_input = (input_signal - self.means) / self.variances
        
        # Scale to 0-1 range
        scaled_input = scaled_input / 2 + 0.5
        
        # project onto unit hypersphere in larger state space
        scaled_input = scaled_input.flatten()
        scaled_input = spherical_transform(scaled_input.reshape(1, len(scaled_input)))

        return scaled_input

    def get_weights(self):
        """ Save the current weights to be used for further runs"""

        return [
            self.sim.signals[self.sim.model.sig[conn]["weights"]]
            for conn in self.conn_learn
        ]

# Intercepts derivation methods ---------------------------------------------------

class Rd(Distribution):

    def __repr__(self):
        return "%s()" % (type(self).__name__)

    def sample(self, n, d=1, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        if d == 1:
            # Tile the points optimally. TODO: refactor
            return np.linspace(1.0 / n, 1, n)[:, None]
        if d is None or not isinstance(d, (int, np.integer)) or d < 1:
            # TODO: this should be raised when the ensemble is created
            raise ValueError("d (%d) must be positive integer" % d)
        return _rd_generate(n, d)

class ScatteredHypersphere(UniformHypersphere):

    def __init__(self, surface, base=Rd()):
        super(ScatteredHypersphere, self).__init__(surface)
        self.base = base

    def __repr__(self):
        return "%s(surface=%r, base=%r)" % (
            type(self).__name__,
            self.surface,
            self.base,
        )

    def sample(self, n, d=1, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        if d == 1:
            return super(ScatteredHypersphere, self).sample(n, d, rng)

        if self.surface:
            samples = self.base.sample(n, d - 1, rng)
            radius = 1.0
        else:
            samples = self.base.sample(n, d, rng)
            samples, radius = samples[:, :-1], samples[:, -1:] ** (1.0 / d)

        mapped = spherical_transform(samples)

        # radius adjustment for ball versus sphere, and a random rotation
        rotation = random_orthogonal(d, rng=rng)
        return np.dot(mapped * radius, rotation)


class SphericalCoords(Distribution):

    def __init__(self, m):
        super(SphericalCoords, self).__init__()
        self.m = m

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.m)

    def sample(self, n, d=None, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        shape = self._sample_shape(n, d)
        y = rng.uniform(size=shape)
        return self.ppf(y)

    def pdf(self, x):
        """Evaluates the PDF along the values ``x``."""
        return np.pi * np.sin(np.pi * x) ** (self.m - 1) / beta(self.m / 2.0, 0.5)

    def cdf(self, x):
        """Evaluates the CDF along the values ``x``."""
        y = 0.5 * betainc(self.m / 2.0, 0.5, np.sin(np.pi * x) ** 2)
        return np.where(x < 0.5, y, 1 - y)

    def ppf(self, y):
        """Evaluates the inverse CDF along the values ``x``."""
        y_reflect = np.where(y < 0.5, y, 1 - y)
        z_sq = betaincinv(self.m / 2.0, 0.5, 2 * y_reflect)
        x = np.arcsin(np.sqrt(z_sq)) / np.pi
        return np.where(y < 0.5, x, 1 - x)


def random_orthogonal(d, rng=None):

    rng = np.random if rng is None else rng
    m = UniformHypersphere(surface=True).sample(d, d, rng=rng)
    u, s, v = svd(m)
    return np.dot(u, v)

def _rd_generate(n, d, seed=0.5):

    def gamma(d, n_iter=20):
        """Newton-Raphson-Method to calculate g = phi_d."""
        x = 1.0
        for _ in range(n_iter):
            x -= (x ** (d + 1) - x - 1) / ((d + 1) * x ** d - 1)
        return x

    g = gamma(d)
    alpha = np.zeros(d)
    for j in range(d):
        alpha[j] = (1 / g) ** (j + 1) % 1

    z = np.zeros((n, d))
    z[0] = (seed + alpha) % 1
    for i in range(1, n):
        z[i] = (z[i - 1] + alpha) % 1

    return z

def spherical_transform(samples):

    samples = np.asarray(samples)
    samples = samples[:, None] if samples.ndim == 1 else samples
    coords = np.empty_like(samples)
    n, d = coords.shape

    # inverse transform method (section 1.5.2)
    for j in range(d):
        coords[:, j] = SphericalCoords(d - j).ppf(samples[:, j])

    # spherical coordinate transform
    mapped = np.ones((n, d + 1))
    i = np.ones(d)
    i[-1] = 2.0
    s = np.sin(i[None, :] * np.pi * coords)
    c = np.cos(i[None, :] * np.pi * coords)
    mapped[:, 1:] = np.cumprod(s, axis=1)
    mapped[:, :-1] *= c
    return mapped