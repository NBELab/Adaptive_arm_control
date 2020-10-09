"""
Written by Dr. Elishai Ezra Tsur 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ 7.9.2020

This work is using Robotis' Dynamixel SDK package (pip install dynamixel-sdk) 
and it is utilized here to control Trossen Robotics' arms. 

Code was tested on the: 
1. 5DOF WidowX 200 (https://www.trossenrobotics.com/widowx-200-robot-arm.aspx)
2. 6DOF ViperX 300 (https://www.trossenrobotics.com/viperx-300-robot-arm-6dof.aspx)
"""

from dynamixel_sdk import *
import time

class RoboticArm:
    
    def __init__ (self, CMD_dict, COM_ID = 'COM5', PROTOCOL_VERSION = 2.0):
        
        self.CMD           = CMD_dict['Real']['CMD']
        self.priority      = CMD_dict['Real']['Priority']
        self.home_position = CMD_dict['Real']['Home']
        
        self.initialize(COM_ID, PROTOCOL_VERSION)
        
    def initialize(self, COM_ID, PROTOCOL_VERSION):  

        self.portHandler   = PortHandler  (COM_ID)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)

        # Open port
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")

        # Broadcast ping the Dynamixel
        ENGINES_list, COMM_result = self.packetHandler.broadcastPing(self.portHandler)
        if COMM_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(COMM_result))
        self.ENGINES = []
        print("Detected Engines :")
        for engine in ENGINES_list:
            print("[ID:%03d] model version : %d | firmware version : %d" % 
                  (engine, ENGINES_list.get(engine)[0], ENGINES_list.get(engine)[1]))           
            if ENGINES_list.get(engine)[0] > 1000: # Checking for identified engines (rather then controllers)
                self.ENGINES.append(engine)

        # Set port baudrate    
        print('Setting baud rate to: 1 Mbps')
        for ID in self.ENGINES:
            self.send_single_cmd(ID, 
                          self.CMD['Baud Rate']['Address'], 
                          self.CMD['Baud Rate']['Value'][1000000])
        
        self.reset_state()
     
    def reset_state (self):
    
        # Setting operation mode to position (default; getting into home position)
        # This command has to be excuted first. Changing modes reset default values.       
        print('Releasing torque')
        self.release_torque()
        
        print('Setting operatio mode to: position')
        for ID in self.ENGINES:
            self.send_single_cmd(ID, 
                               self.CMD['Operating mode']['Address'], 
                               self.CMD['Operating mode']['Value']['Position'])
        
        # Limiting velocity
        print('Limiting velocity to: {}%'.format(100*(self.CMD['Limit velocity']['Value']/885)))
        for ID in self.ENGINES:
            self.send_full_cmd(ID, 
                               self.CMD['Limit velocity']['Address'], 
                               self.CMD['Limit velocity']['Value'])
            
        # Limiting torque
        print('Limiting torque to: {}%'.format(100*(self.CMD['Limit torque']['Value']/1193)))
        for ID in self.ENGINES:
            self.send_half_cmd(ID, 
                               self.CMD['Limit torque']['Address'], 
                               self.CMD['Limit torque']['Value'])
   
    def go_home (self) :
        self.enable_torque()
        print("Setting home position")
        self.set_position(self.home_position)
    
    def reboot (self, IDs = 'all'):
        
        if IDs == 'all':
            IDs = self.ENGINES
        
        for DXL_ID in IDs:
            if DXL_ID < 12: # Assuming less than 12 engines' arm. 
                dxl_comm_result, dxl_error = self.packetHandler.reboot(self.portHandler, DXL_ID)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % self.packetHandler.getRxPacketError(dxl_error))
       
    def enable_torque (self, IDs = 'all'):
        
        if IDs == 'all':
            IDs = self.ENGINES
        
        for DXL_ID in IDs:
            self.send_single_cmd(DXL_ID, 
                          self.CMD['Torque Enable']['Address'], 
                          self.CMD['Torque Enable']['Value']['ON'])  
    
    def release_torque (self, IDs = 'all'):
        
        if IDs == 'all':
            IDs = self.ENGINES
        
        for DXL_ID in IDs:
            self.send_single_cmd(DXL_ID, 
                          self.CMD['Torque Enable']['Address'], 
                          self.CMD['Torque Enable']['Value']['OFF'])
    
    def play (self, sequance, delay, mode='position'):
        
        if mode == 'position':
            for seq in sequance:
                self.set_position(seq)
                time.sleep(delay)
                
        elif mode == 'torque':
            
            targets_TDs = []
            for seq in sequance:
                for ID in seq:
                    if ID not in targets_TDs:
                        targets_TDs.append(ID)
            # Setting targets to torque control mode
            print('Setting operation mode of engines {} to: torque'.format(targets_TDs))
            for ID in targets_TDs:
                self.release_torque(targets_TDs) # Before changing mode, torque have to be released
                self.send_single_cmd(ID, 
                                   self.CMD['Operating mode']['Address'], 
                                   self.CMD['Operating mode']['Value']['Torque'])
                self.enable_torque(targets_TDs)
            
            for seq in sequance:
                self.set_torque(seq)
                time.sleep(delay)
        else:
            print('Working mode is not recognized. Supported modes: position / torque')
            
    def set_torque (self, torque_dict):
        
        for IDs in self.priority:

            for ID in IDs:
                
                if ID not in torque_dict:
                    continue
                
                target_torque = round(torque_dict[ID])
                print('Setting {} to {}'.format(ID, target_torque))
                self.send_half_cmd(ID, 
                                   self.CMD['Goal torque']['Address'], 
                                   target_torque)
                
    
    def set_position (self, position_dict):
        
        for IDs in self.priority:
            
            for ID in IDs:
                
                if ID not in position_dict:
                    continue
                
                target_position = round(position_dict[ID])
                if target_position not in self.CMD['Ranges'][ID]:
                    print('{} is not in range. Engine {} is constrained to {}'.format(
                        position_dict[ID], ID, self.CMD['Ranges'][ID]))
                    continue

                # Multiplying angle by 11.375 to convert to register value
                target_position = round(target_position * 11.375)
                print('Setting {} to {}'.format(ID, target_position/11.375))

                self.send_full_cmd(ID, 
                              self.CMD['Goal Position']['Address'], 
                              target_position)
            
            watchdog    = time.time()
            lapsed_time = 0
            while True:
                lapsed_time = time.time() - watchdog
                conf = [abs(round(position_dict[ID] * 11.375) - self.get_position(ID)) < 20 
                        for ID in IDs]
                if all(conf):
                    break
                if lapsed_time > 2.5:
                    print('watch dog executed')
                    
                    # Set target to current position. 
                    for ID in IDs:
                        self.send_full_cmd(ID, 
                              self.CMD['Goal Position']['Address'], 
                              self.get_position(ID))
                    break
    
    def get_position(self, ID):
        
        dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(
            self.portHandler, ID, self.CMD['Present Position']['Address'])
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))

        return dxl_present_position
    
    def destruct(self):
        self.reboot()
        self.portHandler.closePort()
        
    def send_full_cmd(self, ID, adr, val):
        
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                self.portHandler, ID, adr, val)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            print('ID: {}, Address: {}, value: {}'.format(ID, adr, val))
        else:
            print("[ID:%03d] CMD executed successfully" % ID)
    
    def send_half_cmd(self, ID, adr, val):
        
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
                self.portHandler, ID, adr, val)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            print('ID: {}, Address: {}, value: {}'.format(ID, adr, val))
        else:
            print("[ID:%03d] CMD executed successfully" % ID)
    
    def send_single_cmd(self, ID, adr, val):
        
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, ID, adr, val)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            print('ID: {}, Address: {}, value: {}'.format(ID, adr, val))
        else:
            print("[ID:%03d] CMD executed successfully" % ID)
    
