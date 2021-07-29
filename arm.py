"""
Written by Dr. Elishai Ezra Tsur 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ 25.8.2020

This work is based on ABR's adaptive controller availible at: 
https://github.com/abr/abr_control/tree/master/abr_control 
Using this code is subjected to ABR's licensing

Adaptive control theory is based on:
DeWolf, Travis, Terrence C. Stewart, Jean-Jacques Slotine, and Chris Eliasmith. 
"A spiking neural model of adaptive arm control." 
Proceedings of the Royal Society B: Biological Sciences 283, no. 1843 (2016): 20162134.

Physical simulation is based the MuJoCo simulator (http://www.mujoco.org)
Using the simulator is subject to acquiring a license for MuJoCo (https://www.roboti.us/license.html)

Adaptive control is implemented with the nengo framework (nengo.ai)

Operational space controller is based on:
Khatib, Oussama. 
"A unified approach for motion and force control of robot manipulators: The operational space formulation." 
IEEE Journal on Robotics and Automation 3.1 (1987): 43-53. 

"""
import pickle
from collections import defaultdict

import mujoco_py as mjc
import numpy as np
import glfw
import time

from SNN.ee_sensing import EndEffectorModel
from OSC import OSC
from utilities import euler_from_quaternion
from adaptive_control import DynamicsAdaptation


def calc_error(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


class Simulation:

    def __init__(self, model,  # Instance of the mechanical model of the arm
                 init_angles,  # Initial configuration of the arm null position.
                 target=None,  # Array of target coordinates in reference to the EE position
                 return_to_null=False,  # Return to home position before a approaching a new target
                 th=2e-2,  # Treshold for successful approach to target
                 sim_dt=0.01,  # Simulation time step
                 external_force=None,  # External force field (implemented with scaled gravity)
                 adapt=False,  # Using adaptive controller
                 n_gripper_joints=0, n_neurons=None):  # Number of actuated gripping points

        self.output = []  # tuple of (error,number of iteration) for each experiment
        self.model = model
        self.init_angles = init_angles
        self.target = target
        self.return_to_null = return_to_null
        self.th = th
        self.dt = sim_dt
        self.external_force_field = external_force
        self.n_gripper_joints = n_gripper_joints
        # Initiating the monitor dictionary
        if target is not None:
            self.monitor_dict = {}
            for i, t in enumerate(target):
                self.monitor_dict[i] = {'error': [],  # Delta between current to target location
                                        'ee_simulation': [],  # Position of the end-effector
                                        'ee_integrator': [],  # Position of the end-effector
                                        'q': [],  # Joint angles
                                        'dq': [],  # Joint velocity
                                        'steps': 0,  # number of steps to mitigate the target
                                        'target': t,  # Target coordinates (referenced to the arm null position)
                                        'target_real': 0}  # Target coordinates (referenced to the world)

        # Initialize simulator
        self.simulation = mjc.MjSim(self.model.mjc_model)
        # self.viewer = mjc.MjViewer(self.simulation)

        # Get number of joints
        self.n_joints = int(len(
            self.simulation.data.get_body_jacp('EE')) / 3)  # Jacobian translational component (jacp)

        # Intializing Operational Space Controller (OSC)         
        self.controller = OSC(self)

        # Initiating pose
        self.goto_null_position()
        self.null_position = self.get_ee_position_from_sim()
        self.prev_ee_pos = self.null_position.copy()
        if n_neurons is not None:
            self.ee_model = EndEffectorModel(n_neurons=n_neurons, tau=0.1, transform=50,
                                             height0=self.null_position.copy(), inp_synapse=None
                                             )
        self.from_time = 0
        # Initialize adaptive controller
        self.adaptation = adapt
        if adapt:
            self.adapt_controller = DynamicsAdaptation(
                n_input=10,  # Applying adaptation to the first 5 joints, having 5 angles and 5 velocities
                n_output=5,  # Retrieving 5 adaptive signals for the first 5 joints
                n_neurons=5000,  # Number of neurons for neurons per ensemble
                n_ensembles=5,  # Defining an ensemble for each retrived adaptive signals
                pes_learning_rate=1e-4,  # Learning rate for the PES online learning rule
                means=[  # Scaling the input signals with means / variances of expected values.
                    0.12, 2.14, 1.87, 4.32, 0.59,
                    0.12, -0.38, -0.42, -0.29, 0.36],
                variances=[
                    0.08, 0.6, 0.7, 0.3, 0.6,
                    0.08, 1.4, 1.6, 0.7, 1.2]
            )

    def visualize(self):
        """ visualizing the model with the initial configuration of the arm """

        while True:
            # Exit when the ESC button is pressed
            # if self.viewer.exit:
            #     break

            self.viewer.render()

        glfw.destroy_window(self.viewer.window)

    def simulate(self,
                 steps=None):  # Number of maximum allowable steps for mitigating the target
        """ Simulating the model """

        if self.target is None:
            print('A target eas not defined. Try to visualize instead.')
            return

        # Signifying termination of the simulation 
        breaked = False

        # Iterate over the predefined targets ---------------------------------------------------
        for exp in self.monitor_dict:
            sum_sqr_err = 0
            # Terminate the simulation if it was signaled to using the ESC key
            if breaked:
                break

            # Retrieving the position of the target in world's coordinates
            target = self.null_position + self.monitor_dict[exp]['target']
            self.monitor_dict[exp]['target_real'] = np.copy(target[:3])

            # Setting the location of target (sphere; defined in the XML model) in the simulation
            self.simulation.data.set_mocap_pos("target", target)

            # Keeping track of the simulation's number of steps
            step = 0

            # Initializing error
            error = float("inf")

            while True:  # Execute simulation -----------------------------------------------------

                # Breaking conditions ------------------------------------------------------------

                # Keeping track of the steps and moving to the next target if exceeding limit
                step += 1
                if steps is not None:
                    if step > steps:
                        self.monitor_dict[exp]['steps'] = step
                        break

                # Terminate simulation with ESC key
                # if self.viewer.exit:
                #     breaked = True
                #     break

                # Terminate, or move to the next target when the EE is within 
                # the threshold value of the target
                if error < 1e-2:
                    self.monitor_dict[exp]['steps'] = step
                    if self.return_to_null:
                        self.goto_null_position()
                    break  #

                # Calculating control signals ----------------------------------------------------

                # Force array which will be sent to the arm
                u = np.zeros(self.model.n_joints)

                # Retrieve the joint angle values and velocities
                position, velocity = (self.get_angles(), self.get_velocity())

                # Request the OSC to generate control signals to actuate the arm

                u = self.controller.generate(position, velocity, target, self.get_ee_position())

                # Converting the retrieved dictionary to force arrays to be send to the arm
                # Only the first 5 actuators are activated. 
                # The six'th actuator controls the EE orientation.
                position_array = [np.copy(position[i]) for i in range(5)]
                velocity_array = [np.copy(velocity[i]) for i in range(5)]

                # If adaptation mode is on, that retireve the adapt signals
                if self.adaptation:
                    u_adapt = np.zeros(self.model.n_joints)
                    # Retrieveing the adapt signals from the adaptive controller
                    u_adapt[:5] = self.adapt_controller.generate(
                        # 10 inputs constituting the arm's joints' angles and velocities
                        input_signal=np.hstack((position_array, velocity_array)),
                        # Training signal for the controller. 
                        # Training signal is the actuation values retrieved before, 
                        # without the gravitational force field. 
                        training_signal=np.array(self.controller.training_signal[:5]),
                    )
                    # Update the control signal with adaptation
                    u += u_adapt

                # Adding an external force field to the arm, if such was defined
                if self.external_force_field is not None:
                    extra_gravity = self.get_gravity_bias() * self.external_force_field
                    u += extra_gravity

                # Accounting for the not-moving grippers (to adapt dimensions)
                u = np.hstack((u, np.zeros(self.n_gripper_joints)))

                # Actuating the arm, calculate error and update viewer ---------------------------
                self.send_forces(u)
                # self.viewer.render()

                # retrieve the position of the arm follow actuation
                _ee_position = self.get_ee_position_from_sim()

                # ee velocity from simulation (outside sensor)
                if hasattr(self, 'ee_model'):
                    ee_vel = _ee_position - self.prev_ee_pos
                    self.ee_model.update(ee_vel)
                    self.prev_ee_pos = _ee_position.copy()

                # Calculate error as the distance between the target and the position of the EE
                ee_xyz = self.get_ee_position()
                error = calc_error(target, ee_xyz)
                sum_sqr_err += error
                # Monitoring  --------------------------------------------------------------------

                self.monitor_dict[exp]['error'].append(np.copy(error))  # Error step
                self.monitor_dict[exp]['ee_simulation'].append(
                    np.copy(self.get_ee_position_smooth()))  # Position of the EE
                self.monitor_dict[exp]['q'].append(np.copy(position))  # Joints' angles
                self.monitor_dict[exp]['dq'].append(np.copy(velocity))  # Joints' velocities
            # y_time, x = self.ee_model.get_xy()
            # self.monitor_dict[exp]['ee_integrator'] = x[self.from_time:]
            # self.from_time = len(y_time)  # divide position to experiments
            self.output.append((sum_sqr_err, step))
        # End of simulation ----------------------------------------------------------------------
        time.sleep(1.5)
        # glfw.destroy_window(self.viewer.window)
        return self.output

    # Arm actuation methods ----------------------------------------------------------------------

    def goto_null_position(self):
        """ Return arm null position, specified by init_angles """

        self.send_target_angles(self.init_angles)

    def send_target_angles(self, q):
        """ Move the arm to the specified joint configuration """

        for j in q:
            self.simulation.data.qpos[self.model.joint_dict[j]['position_address']] = q[j]
        self.simulation.forward()  # Compute forward kinematics

    def send_forces(self, u):
        """ Apply the specified torque to the robot joints """

        # Setting the forces to the specified joints (assuming array ordering)
        self.simulation.data.ctrl[:] = u[:]

        # move simulation ahead one time step
        self.simulation.step()

        # Update position of hand object
        self.simulation.data.set_mocap_pos("hand", self.get_ee_position_from_sim())

        # Update orientation of hand object
        quaternion = np.copy(self.simulation.data.get_body_xquat("EE"))
        self.simulation.data.set_mocap_quat("hand", quaternion)

    # Retrieve arm properties actuation methods --------------------------------------------------

    def get_ee_position(self):
        """ Retrieve the position of the End Effector (EE) """
        if hasattr(self, 'ee_model'):
            return self.ee_model.get_curr_pos()
        return self.get_ee_position_from_sim()

    def get_ee_position_smooth(self):
        """ Retrieve the position of the End Effector (EE) """
        if hasattr(self, 'ee_model'):
            return self.ee_model.get_curr_pos_smooth()
        return self.get_ee_position_from_sim()

    def get_ee_position_from_sim(self):

        return np.copy(self.simulation.data.get_body_xpos('EE'))

    def get_angles(self):
        """ Returns joint angles [rad] """

        q = {}
        for joint in self.model.joint_dict:
            q[joint] = np.copy(self.simulation.data.qpos[
                                   self.model.joint_dict[joint]['position_address']])
        return q

    def get_velocity(self):
        """ Returns joint velocity [rad/sec] """

        v = {}
        for joint in self.model.joint_dict:
            v[joint] = np.copy(self.simulation.data.qvel[
                                   self.model.joint_dict[joint]['velocity_address']])
        return v

    def get_target(self):
        """ Returns the position and orientation of the target """

        xyz_target = self.simulation.data.get_body_xpos("target")
        quat_target = self.simulation.data.get_body_xquat("target")
        euler_angles = euler_from_quaternion(quat_target)
        return np.hstack([np.copy(xyz_target), np.copy(euler_angles)])

    def get_Jacobian(self):
        """ Returns the Jacobian of the arm (from the perspective of the EE) """

        _J3NP = np.zeros(3 * self.n_joints)
        _J3NR = np.zeros(3 * self.n_joints)
        _J6N = np.zeros((6, self.model.n_joints))

        joint_dyn_addrs = np.array((list(self.model.joint_dict.keys())))

        # Position and rotation Jacobians are 3 x N_JOINTS
        jac_indices = np.hstack(
            [joint_dyn_addrs + (ii * self.n_joints) for ii in range(3)])

        mjc.cymj._mj_jacBodyCom(
            self.model.mjc_model, self.simulation.data,
            _J3NP, _J3NR, self.model.mjc_model.body_name2id('EE')
        )

        # get the position / rotation Jacobian hstacked (1 x N_JOINTS*3)
        _J6N[:3] = _J3NP[jac_indices].reshape((3, self.model.n_joints))
        _J6N[3:] = _J3NR[jac_indices].reshape((3, self.model.n_joints))

        return np.copy(_J6N)

    def get_inertia_matrix(self):
        """ Returns the inertia matrix of the arm """

        _MNN = np.zeros(self.n_joints ** 2)

        joint_dyn_addrs = np.array((list(self.model.joint_dict.keys())))
        self.M_indices = [
            ii * self.n_joints + jj
            for jj in joint_dyn_addrs
            for ii in joint_dyn_addrs
        ]

        # stored in mjData.qM, stored in custom sparse format,
        # convert qM to a dense matrix with mj_fullM
        mjc.cymj._mj_fullM(self.model.mjc_model, _MNN, self.simulation.data.qM)

        M = _MNN[self.M_indices]
        M = M.reshape((self.model.n_joints, self.model.n_joints))
        return np.copy(M)

    def get_gravity_bias(self):
        """ Returns the effects of Coriolis, centrifugal, and gravitational forces """

        joint_dyn_addrs = np.array((list(self.model.joint_dict.keys())))
        g = -1 * self.simulation.data.qfrc_bias[joint_dyn_addrs]
        return g

    # Monitoring methods -------------------------------------------------------------------------

    def show_monitor(self):
        """
         Display monitored motion and performance of the arm
         every experiment is a row and we have position (simulation), position (integrator)
         expected: all lines merge in the process.
        """

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d
        from scipy.ndimage import gaussian_filter
        # ratio numbers
        n_experiments = len(self.monitor_dict)
        fig = plt.figure()
        rows = int(np.sqrt(n_experiments))
        cols = int(np.ceil(n_experiments / rows))

        # For each specified target 
        for index, exp in enumerate(self.monitor_dict, start=1):
            # position of ee and target
            ax = fig.add_subplot(rows, cols, index, projection='3d')
            sim_ee = np.array(self.monitor_dict[exp]['ee_simulation'])

            # with open(f'simpos_exp_{index}.pkl', 'wb') as fp:
            #     pickle.dump(sim_ee, fp)
            with open(f'simpos_exp_{index}.pkl', 'rb') as fp:
                fk_pos = pickle.load(fp)

            real_target = self.monitor_dict[exp]['target_real']

            ax.set_title(
                "End-Effector Trajectory (Target:{})".format(index))

            # integrator_ee = np.array(self.monitor_dict[exp]['ee_integrator'])
            ax.plot(*sim_ee.T, label='integrator')
            ax.plot(*fk_pos.T, label='simulation')
            # ax.plot(*integrator_ee.T, label='integrator')
            ax.scatter(*real_target, label="target", c="r")
            ax.legend()
            # difference in simulation position and integrator value in the second row
            # ax = fig.add_subplot(2, n_experiments, index + n_experiments)
            # ax.set_title("End-Effector position difference (sim vs. integrator)".format(index))
            # ax.plot(np.linalg.norm(sim_ee - integrator_ee, axis=1), label='norm2 difference')
            # ax.legend()

        plt.show()
