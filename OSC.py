"""
Written by Dr. Elishai Ezra Tsur 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ 25.8.2020

Operational space controller is based on:
Khatib, Oussama. 
"A unified approach for motion and force control of robot manipulators: The operational space formulation." 
IEEE Journal on Robotics and Automation 3.1 (1987): 43-53.

"""

import numpy as np

class OSC:
    
    def __init__(self, simulation, **kwarg):
        
        self.simulation = simulation
        
        _DEFAULTS =  {'Kv': 20, 'Kp': 200, 'Ko': 200, 'Ki': 0, 'vmax': [0.5, 0]}              
        self.control_dict = _DEFAULTS          
        
        for param in kwarg:
            self.control_dict[param] = kwarg[param]
        
        self.control_dict['n_joints'] = self.simulation.model.n_joints
                      
        self.control_dict['task_space_gains'] = np.array([self.control_dict['Kp']] * 3 + [self.control_dict['Ko']] * 3)           
        self.control_dict['lamb'] = self.control_dict['task_space_gains'] / self.control_dict['Kv']
        self.control_dict['scale_xyz'] = self.control_dict['sat_gain_xyz'] = \
                        self.control_dict['vmax'][0] / self.control_dict['Kp'] * self.control_dict['Kv']
        self.control_dict['scale_abg'] = self.control_dict['sat_gain_abg'] = \
                        self.control_dict['vmax'][1] / self.control_dict['Ko'] * self.control_dict['Kv']
    
    def generate (self, q, dq, target):
                      
        target_velocity = np.zeros(len(q.items()))
                                   
        # isolate rows of Jacobian corresponding to controlled task space DOF
        # Particularly, shosing x,y,x among [x, y, z, alpha, beta, gamma] 
        J = self.simulation.get_Jacobian()
        control_dof = [True, True, True, False, False, False]
        J = J[control_dof]

        # Getting the inertia matrix                           
        M = self.simulation.get_inertia_matrix()  # inertia matrix in joint space
        Mx, M_inv = self._Mx(M=M, J=J)     # inertia matrix in task space
        
        # calculate the desired task space forces 
        u_task = np.zeros(6)

        # position controlled (orientation control TBA)
        xyz = self.simulation.simulation.data.get_body_xpos("EE")
        u_task[:3] = xyz - target[:3]

        # task space integrated error 
        integrated_error = np.zeros(6)
        if self.control_dict['Ki'] != 0:
            integrated_error += u_task
            u_task += self.control_dict['Ki'] * integrated_error

        # if max task space velocities specified, apply velocity limiting
        if self.control_dict['vmax'] is not None:
            u_task = self._velocity_limiting(u_task)
        else:
            # otherwise apply specified gains
            u_task *= self.control_dict['task_space_gains']
            
        # As there's no target velocity in task space,
        # compensate for velocity in joint space (more accurate)
        u = np.zeros(self.control_dict['n_joints'])
        dq_vector = [float(dq[i]) for i in range(self.control_dict['n_joints'])]
        u = -1 * self.control_dict['Kv'] * np.dot(M, dq_vector)

        # isolate task space forces corresponding to controlled DOF
        u_task = u_task[control_dof]

        # transform task space control signal into joint space ----------------
        u -= np.dot(J.T, np.dot(Mx, u_task))

        # store the current control signal u for training in case
        # dynamics adaptation signal is being used
        # NOTE: do not include gravity or null controller in training signal
        self.training_signal = np.copy(u)
        
        # add in gravity term in joint space ----------------------------------
        u -= self.simulation.get_gravity_bias()

        
        return u
      
    def _Mx(self, M, J, threshold=1e-3):
        """ Generate the task-space inertia matrix """

        # calculate the inertia matrix in task space
        M_inv = np.linalg.inv(M)
        Mx_inv = np.dot(J, np.dot(M_inv, J.T))
        if abs(np.linalg.det(Mx_inv)) >= threshold:
            # do the linalg inverse if matrix is non-singular
            # because it's faster and more accurate
            Mx = np.linalg.inv(Mx_inv)
        else:
            # using the rcond to set singular values < thresh to 0
            # singular values < (rcond * max(singular_values)) set to 0
            Mx = np.linalg.pinv(Mx_inv, rcond=threshold * 0.1)

        return Mx, M_inv
    
    def _velocity_limiting(self, u_task): 
        """ Scale the control signal to limit the velocity of the arm """
        
        norm_xyz = np.linalg.norm(u_task[:3])
        norm_abg = np.linalg.norm(u_task[3:])
                 
        scale = np.ones(6)
        if norm_xyz > self.control_dict['sat_gain_xyz']:
            scale[:3] *= self.control_dict['scale_xyz'] / norm_xyz
        if norm_abg > self.control_dict['sat_gain_abg']:
            scale[3:] *= self.control_dict['scale_abg'] / norm_abg

        return self.control_dict['Kv'] * scale * self.control_dict['lamb'] * u_task