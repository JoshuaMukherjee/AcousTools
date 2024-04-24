from ctypes import CDLL, POINTER
import ctypes

import torch, os

from acoustools.Utilities import get_convert_indexes

class LevitatorController():
    '''
     Class to enable the manipulation of an acoustic levitator from python. 
    '''

    def __init__(self, bin_path = None, ids = (1000,999), matBoardToWorld=None, print_lines=False):
        '''
        Creates the controller\\
        `bin_path`: The path to the binary files needed. If none will use files contained in AcousToosl. NOTE WHEN SET TO NONE THIS CHANGES THE CURRENT WORKING DIRECTORY AND THEN CHANGES IT BACK Default: None.\\
        `ids`: IDs of boards. Default `(1000,999)`\\
        `matBoardToWorld`: Matric defining the mapping between simulated and real boards. When None uses a default setting. Default None.\\
        `print_lines`: If False supresses some print messages
        '''

        if bin_path is None:
            self.bin_path = os.path.dirname(__file__)+"/../../bin/x64/"
        
        cwd = os.getcwd()
        os.chdir(self.bin_path)
        print(os.getcwd())
        self.levitatorLib = CDLL(self.bin_path+'Levitator.dll')

        self.ids = (ctypes.c_int * 2)(*ids)
        self.board_number = len(ids)

        if matBoardToWorld is None:
            self.matBoardToWorld =  (ctypes.c_float * (16*self.board_number)) (
                
                -1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0.24,
                0, 0, 0, 1,

                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
                
            )
        else:
            self.matBoardToWorld =  (ctypes.c_float * (16*self.board_number))(matBoardToWorld)

        self.levitatorLib.connect_to_levitator.restype = ctypes.c_void_p
        self.controller = self.levitatorLib.connect_to_levitator(self.ids,self.matBoardToWorld,self.board_number,print_lines)


        os.chdir(cwd)

        self.IDX = get_convert_indexes()
    
    
    def send_message(self, phases, amplitudes=None, relative_amplitude=1, num_geometries = 1, sleep_ms = 0):
        '''
        RECCOMENDED NOT TO USE - USE `levitate` INSTEAD\\
        sends messages to levitator
        '''
        self.levitatorLib.send_message.argtypes = [ctypes.c_void_p,POINTER(ctypes.c_float), POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_int, ctypes.c_int]
        self.levitatorLib.send_message(self.controller,phases,amplitudes,relative_amplitude,num_geometries, sleep_ms)
    
    def disconnect(self):
        '''
        Disconnects the levitator
        '''
        self.levitatorLib.disconnect.argtypes = [ctypes.c_void_p]
        self.levitatorLib.disconnect(self.controller)
    
    def turn_off(self):
        '''
        Turns of all transducers
        '''
        self.levitatorLib.turn_off.argtypes = [ctypes.c_void_p]
        self.levitatorLib.turn_off(self.controller)

    def levitate(self, phases, amplitudes=None, relative_amplitude=1, permute=True, sleep_ms = 0):
        '''
        Send a single phase map to the levitator - This is the reccomended function to use as will deal with dtype conversions etc\\
        `phases`: `Torch.Tensor` of phases or list of `Torch.Tensor` of phases, expects a batched dimension in dim 0. If phases is complex then ` phases = torch.angle(phases)` will be run, else phases left as is\\
        `amplitudes`: Optional `Torch.Tensor` of amplitudes, same shape as `phases`\\
        `relative_amplitude`: Single value [0,1] to set amplitude to. Default 1\\
        `permute`: Convert between acoustools transducer order and OpenMPD. Default True.\\
        `sleep_ms`: Time to wait between frames in ms.
        '''
        to_output = []

        if type(phases) is list:
            num_geometries = len(phases)
            for phases_elem in phases:

                if permute:
                    phases_elem = phases_elem[:,self.IDX]

                if torch.is_complex(phases_elem):
                    phases_elem = torch.angle(phases_elem)
        
                to_output = to_output + phases_elem.squeeze().cpu().detach().tolist()
        else:
            num_geometries = 1
            if permute:
                    phases = phases[:,self.IDX]

            if torch.is_complex(phases):
                    phases = torch.angle(phases)
            to_output = phases[0].squeeze().cpu().detach().tolist()

        phases = (ctypes.c_float * (256*self.board_number *num_geometries))(*to_output)

        if amplitudes is not None:
            amplitudes = (ctypes.c_float * (256*self.board_number*num_geometries))(*amplitudes)

        relative_amplitude = ctypes.c_float(relative_amplitude)

        self.send_message(phases, amplitudes, relative_amplitude, num_geometries,sleep_ms=sleep_ms)