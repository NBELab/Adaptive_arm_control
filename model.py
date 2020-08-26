"""
Written by Dr. Elishai Ezra Tsur 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ 25.8.2020

Physical simulation is based the MuJoCo simulator (http://www.mujoco.org)
Using the simulator is subject to acquiring a license for MuJoCo (https://www.roboti.us/license.html)
"""

from os import path
import numpy as np

import mujoco_py as mjc

class Model:
    
    def __init__(self, xml_specification, mesh_specification=None):
        
        self.xml_specification  = xml_specification
        self.mesh_specification = mesh_specification
        
        if not path.isfile(self.xml_specification):
            raise Exception('Missing XML specification at: {}'.format(self.xml_specification))
        
        if mesh_specification is not None:
            if not path.isdir(self.mesh_specification):
                raise Exception('Missing mesh specification at: {}'.format(self.mesh_specification))
        
        print('Arm model is specified at: {}'.format(self.xml_specification))
        
        try:
            self.mjc_model = mjc.load_model_from_path(self.xml_specification)
        except:
            raise Exception('Mujoco was unable to load the model')
                
        # Initializing joint dictionary
        joint_ids, joint_names = self.get_joints_info()
        joint_positions_addr   = [self.mjc_model.get_joint_qpos_addr(name) for name in joint_names]
        joint_velocity_addr    = [self.mjc_model.get_joint_qvel_addr(name) for name in joint_names]
        self.joint_dict        = {} 
        for i, ii in enumerate(joint_ids):
            self.joint_dict[ii] = {'name': joint_names[i], 
                                   'position_address': joint_positions_addr[i], 
                                   'velocity_address': joint_velocity_addr[i]}

        if not np.all(np.array(self.mjc_model.jnt_type)==3): # 3 stands for revolute joint
            raise Exception('Revolute joints are assumed')
            
        self.n_joints = len(self.joint_dict.items())
    
    def visualize(self):
        
        simulation = mjc.MjSim(self.mjc_model)
        viewer     = mjc.MjViewer(simulation)
        
        while True:
            
            if viewer.exit:
                break
                
            viewer.render()
            
        glfw.destroy_window(viewer.window)            
            
    def get_joints_info(self):
        
        model = self.mjc_model
        joint_ids = []
        joint_names = []
        body_id = model.body_name2id("EE")
        while model.body_parentid[body_id] != 0:
            jntadrs_start = model.body_jntadr[body_id]
            tmp_ids = []
            tmp_names = []
            for ii in range(model.body_jntnum[body_id]):
                tmp_ids.append(jntadrs_start + ii)
                tmp_names.append(model.joint_id2name(tmp_ids[-1]))
            joint_ids += tmp_ids[::-1]
            joint_names += tmp_names[::-1]
            body_id = model.body_parentid[body_id]
        joint_names = joint_names[::-1]
        joint_ids = np.array(joint_ids[::-1])

        return joint_ids, joint_names  

