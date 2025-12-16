from ctypes import CDLL, POINTER
import ctypes

import torch, os

from acoustools.Utilities import get_convert_indexes
from torch import Tensor


class LevitatorController():
    '''
     Class to enable the manipulation of an OpenMPD style acoustic levitator from python. 
    '''

    def __init__(self, bin_path:str|None = None, ids:tuple[int] = (1000,999), matBoardToWorld:list[int]|None=None, 
                 print_lines:bool=False):
        '''
        Creates the controller\n
        ```python
        from acoustools.Levitator import LevitatorController
        from acoustools.Utilities import create_points, add_lev_sig, propagate_abs
        from acoustools.Solvers import wgs

        lev = LevitatorController()

        p = create_points(1,1,x=0,y=0,z=0)
        x = wgs(p)
        print(propagate_abs(x,p))
        x = add_lev_sig(x)

        lev.levitate(x)
        print('Levitating...')
        input()
        print('Stopping...')
        lev.disconnect()
        print('Stopped')

        ```
        THIS CHANGES THE CURRENT WORKING DIRECTORY AND THEN CHANGES IT BACK \n
        :param bin_path: The path to the binary files needed. If `None` will use files contained in AcousTools. Default: None.
        :param ids: IDs of boards. Default `(1000,999)`. For two board setup will be (Top, Bottom) if `-1` then all messages will be ignored. Use when testing code but no device is conncted 
        :param matBoardToWorld: Matric defining the mapping between simulated and real boards. When `None` uses a default setting. Default `None`.
        :param print_lines: If False supresses some print messages
        '''
        
        self.mode = 1
        '''
        @private
        '''
        if type(ids) == int:
            ids = (ids,)

        if ids[0] == -1:
            self.mode = 0
            print('Virtual Levitator mode - no messages will be sent')
        else:
            if bin_path is None:
                self.bin_path = os.path.dirname(__file__)+"/../../bin/x64/"
            
            cwd = os.getcwd()
            os.chdir(self.bin_path)
            files = os.listdir()
            
            for id in ids:
                if 'board_'+str(id)+'.pat' not in files:
                    data_file = open('board_master.pat','r')
                    data = data_file.read()

                    file = open('board_'+str(id)+'.pat','w')
                    data_id = data.replace('<XXXXXXXX>',str(id))
                    file.write(data_id)
                    file.close()
                    data_file.close()


            print(os.getcwd())
            self.levitatorLib = CDLL(self.bin_path+'Levitator.dll')

            self.board_number = len(ids)
            self.ids = (ctypes.c_int * self.board_number)(*ids)

            if matBoardToWorld is None:
                if self.board_number == 2:
                    self.matBoardToWorld =  (ctypes.c_float * (16*self.board_number)) (
                        1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1,

                        1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1

                        
                    )
                elif self.board_number == 1:
                     self.matBoardToWorld =  (ctypes.c_float * (16*self.board_number)) (
                        1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1
                        )
                else:
                    raise ValueError('For number of boards > 2, matBoardToWorld shouldnt be None')

            else:
                self.matBoardToWorld =  (ctypes.c_float * (16*self.board_number))(*matBoardToWorld)
            

            self.levitatorLib.connect_to_levitator.argtypes =  [POINTER(ctypes.c_int), POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_bool]
            self.levitatorLib.connect_to_levitator.restype = ctypes.c_void_p
            self.controller = self.levitatorLib.connect_to_levitator(self.ids,self.matBoardToWorld,self.board_number,print_lines)

            os.chdir(cwd)

            self.IDX = get_convert_indexes(256*self.board_number).cpu().detach()
    
    
    def send_message(self, phases, amplitudes=None, relative_amplitude=1, num_geometries = 1, sleep_ms = 0, loop=False, num_loops = 0):
        '''
        RECCOMENDED NOT TO USE - USE `levitate` INSTEAD\\
        @private
        sends messages to levitator
        '''
        if self.mode:
            self.levitatorLib.send_message.argtypes = [ctypes.c_void_p,POINTER(ctypes.c_float), POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            self.levitatorLib.send_message(self.controller,phases,amplitudes,relative_amplitude,num_geometries, sleep_ms, loop, num_loops)

    def disconnect(self):
        '''
        Disconnects the levitator
        '''
        if self.mode:
            self.levitatorLib.disconnect.argtypes = [ctypes.c_void_p]
            self.levitatorLib.disconnect(self.controller)
    
    def turn_off(self):
        '''
        Turns of all transducers
        '''
        if self.mode:
            self.levitatorLib.turn_off.argtypes = [ctypes.c_void_p]
            self.levitatorLib.turn_off(self.controller)
        
    def set_frame_rate(self, frame_rate:int):
        '''
        Set a new framerate
        :param frame_rate: The new frame rate to use. Note OpenMPD cannot use framerates below 157Hz
        '''
        if self.mode:
            self.levitatorLib.set_new_frame_rate.argtypes = [ctypes.c_void_p, ctypes.c_int]
            new_frame_rate = self.levitatorLib.set_new_frame_rate(self.controller, frame_rate)

    def levitate(self, hologram:list[Tensor]|Tensor, relative_amplitude:int=-1, permute:bool=True, sleep_ms:float = 0, loop:bool=False, num_loops:int=0):
        '''
        Send a single phase map to the levitator - This is the recomended function to use as will deal with dtype conversions etc
        :param hologram: `Torch.Tensor` of phases or list of `Torch.Tensor` of phases, expects a batched dimension in dim 0. If phases is complex then ` phases = torch.angle(hologram)` will be run for phase and ` amp = torch.abs(hologram)` for amplitude, else phases left as is
        :param relative_amplitude: Single value [0,1] or -1 to set amplitude to. If -1 will ignore Default -1
        :param permute: Convert between acoustools transducer order and OpenMPD. Default True.
        :param sleep_ms: Time to wait between frames in ms.
        :param loop: If True will restart from the start of phases, default False
        :param num_loops: A set number of times to repeat the phases
        '''


        if self.mode:
            to_output = []
            to_output_amplitudes = []

            if type(hologram) is Tensor and hologram.shape[0] > 1:
                holos = []
                for h in hologram:
                    holos.append(h.unsqueeze(0).cpu().detach())
                hologram = holos

            if type(hologram) is list:
                #chunk this up - blocks of 32....
                num_geometries = len(hologram)
                for phases_elem in hologram:
                    phases_elem = phases_elem.cpu().detach()

                    if permute:
                        phases_elem = phases_elem[:,self.IDX]

                    if torch.is_complex(phases_elem):
                        amp_elem = torch.abs(phases_elem)
                        phases_elem = torch.angle(phases_elem)
                        
                    else:
                        amp_elem = torch.ones_like(phases_elem)
            
                    to_output = to_output + phases_elem.squeeze().tolist()
                    to_output_amplitudes = to_output_amplitudes + amp_elem.squeeze().tolist()
            else:
                num_geometries = 1
                if permute:
                        hologram = hologram.cpu().detach()
                        hologram = hologram[:,self.IDX]

                if torch.is_complex(hologram):
                        amp = torch.abs(hologram)
                        hologram = torch.angle(hologram)
                else:
                        amp = torch.ones_like(hologram)
                to_output = hologram[0].squeeze().tolist()
                to_output_amplitudes = amp[0].squeeze().tolist()


            phases = (ctypes.c_float * (256*self.board_number *num_geometries))(*to_output)
           

            if relative_amplitude == -1:
                amplitudes = (ctypes.c_float * (256*self.board_number*num_geometries))(*to_output_amplitudes)
            else:
                amplitudes = None
                relative_amplitude = ctypes.c_float(relative_amplitude)
            
            self.send_message(phases, amplitudes, 0, num_geometries,sleep_ms=sleep_ms,loop=loop,num_loops=num_loops)