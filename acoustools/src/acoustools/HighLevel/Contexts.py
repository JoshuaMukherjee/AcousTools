from acoustools.Solvers import wgs
from acoustools.Utilities import create_points, TRANSDUCERS, transducers
from acoustools.Visualiser import Visualise, ABC
from acoustools.Levitator import LevitatorController

from acoustools.Mesh import load_scatterer, translate
from acoustools.BEM import get_cache_or_compute_H, propagate_BEM_pressure, compute_E

import torch
from torch import Tensor

from typing import Iterable


class AcousToolsContext():
    def __init__(self, origin:Iterable|Tensor=(0,0,0)):

        if type(origin) is not Tensor:
            self.origin = create_points(1,1,origin[0], origin[1], origin[2])

        self.lev = None

        self.solution=None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.lev is not None:
            self.lev.disconnect()



class TopBottomContext(AcousToolsContext):

    def __init__(self, origin:Iterable|Tensor=(0,0,0), default_solver=wgs):

        super().__init__(origin)

        self.solver = default_solver
        self.board = TRANSDUCERS


    def create_focus(self, location = None):
        if location is None:
            location = self.origin
        
        if type(location) is not Tensor:
            location = create_points(1,1,location[0], location[1], location[2])

        self.solution = self.solver(location, board = self.board)
    
    def visualise(self, size=0.03, origin:Tensor=None, plane:str='xz'):
        abc = ABC(size, origin=origin, plane=plane)
        Visualise(*abc, self.solution)


    def send_solution(self, ids=(1000,999)):
        if self.lev is None: self.lev = LevitatorController(ids=ids)

        self.lev.levitate(self.solution)



class TopWithReflectorContext(AcousToolsContext):
    def __init__(self, origin:Iterable|Tensor=(0,0,0), default_solver=wgs, path='../BEMMedia/', reflector_path = './flat-lam2.stl', height = 0.12):

        super().__init__(origin)

        self.solver = default_solver
        self.board = transducers(16, z=height/2)

        self.path = path
        self.reflector = load_scatterer(path + reflector_path)
        translate(self.reflector, dz=-1*height/2)


        self.H = get_cache_or_compute_H(self.reflector, self.board, path=path)

        
    def create_focus(self, location = None):
        if location is None:
            location = self.origin
        
        if type(location) is not Tensor:
            location = create_points(1,1,location[0], location[1], location[2])


        E = compute_E(self.reflector, location, self.board, path=self.path, H=self.H)
        self.solution = self.solver(location, board = self.board, A=E)

    
    def visualise(self, size=0.03, origin:Tensor=None, plane:str='xz'):
        abc = ABC(size, origin=origin, plane=plane)
        Visualise(*abc, self.solution, colour_functions=[propagate_BEM_pressure], colour_function_args=[{'path':self.path, "H":self.H, 'board':self.board, 'scatterer':self.reflector}])

    def send_solution(self, ids=(1000)):
        if self.lev is None: self.lev = LevitatorController(ids=ids)

        self.lev.levitate(self.solution)



class BottomWithReflectorContext(TopWithReflectorContext):
    def __init__(self, origin:Iterable|Tensor=(0,0,0), default_solver=wgs, path='../BEMMedia/', reflector_path = './flat-lam2.stl', height = 0.12):

        super().__init__(origin, default_solver, path, reflector_path, height)

        self.board = transducers(16, z=-1*height/2)
        
        self.reflector = load_scatterer(path + reflector_path)
        translate(self.reflector, dz=height/2)


        self.H = get_cache_or_compute_H(self.reflector, self.board, path=path)