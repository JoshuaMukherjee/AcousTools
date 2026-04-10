from acoustools.Solvers import wgs
from acoustools.Utilities import create_points, TRANSDUCERS, transducers
from acoustools.Visualiser import Visualise, ABC
from acoustools.Levitator import LevitatorController

from acoustools.Mesh import load_scatterer, translate
from acoustools.BEM import get_cache_or_compute_H, propagate_BEM_pressure, compute_E

import torch
from torch import Tensor

from typing import Iterable, Self

from vedo.mesh import Mesh

from vedo.mesh import Mesh


class AcousToolsContext():

    def __init__(self, origin:Iterable|Tensor=(0,0,0)) -> None:
        '''
        Base Class for AcousTools Context Objects \n
        :param origin: The position of the origin for the setup \n

        Should be used in a `with` block
        '''

        if type(origin) is not Tensor:
            self.origin: Tensor = create_points(1,1,origin[0], origin[1], origin[2])

        self.lev = None

        self.solution=None
    
    def __enter__(self) -> Self:
        return self
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.lev is not None:
            self.lev.disconnect()



class TopBottomContext(AcousToolsContext):


    def __init__(self, origin:Iterable|Tensor=(0,0,0), default_solver=wgs) -> None:
        '''
        AcousTools Context Objects  for 2 boards with a seperation of 23.65cm\n
        :param origin: The position of the origin for the setup \n
        :param default_solver: The solver to use when creating a focus, default wgs

        Should be used in a `with` block eg
        ```
        from acoustools.HighLevel import TopBottomContext

        with TopBottomContext() as ctx:
            ctx.create_focus()
            ctx.send_solution()
        ```
    '''        

        super().__init__(origin)

        self.solver = default_solver
        self.board: Tensor = TRANSDUCERS


    def create_focus(self, location:None|Tensor|Iterable = None) -> None:
        '''
        Create a focus at `location` using the default solver defined in class creation and stores the result in the object. \n
        The user does not need to know how the focus is created or what the ouput of the computation is \n
        Should be called before `send_solution`  or `visualise`
        :param location: The location of the focus
        '''

        if type(location) not in (None, Tensor, Iterable):
            raise TypeError("Locations should be one of None, Tensor, Iterable")

        if location is None:
            location: Tensor = self.origin
        
        if type(location) is not Tensor:
            location: Tensor = create_points(1,1,location[0], location[1], location[2])

        self.solution = self.solver(location, board = self.board)
    
    def visualise(self, size:float=0.03, origin:Tensor=None, plane:str='xz') -> None:
        '''
        Visualises the sound field that has been previously computed, see `acoustools.Visualise.Visualiser` for parameter details \n
        :param size: Size of the visualisation window from the origin (window will be 2*size)
        :param origin: Central point of the visualisation
        :param plane: Plane to visulaise in
        '''
        if self.solution is None:
            raise ValueError("No solution found, call `create_focus` before use")
        abc: tuple[Tensor] = ABC(size, origin=origin, plane=plane)
        Visualise(*abc, self.solution)


    def send_solution(self, ids=(1000,999)) -> None:
        '''
        Sends the solution to a connected physical device
        :param ids: The ids for the device connected
        '''
        if self.solution is None:
            raise ValueError("No solution found, call `create_focus` before use")
        
        if self.lev is None: self.lev = LevitatorController(ids=ids)

        self.lev.levitate(self.solution)



class TopWithReflectorContext(AcousToolsContext):


    def __init__(self, origin:Iterable|Tensor=(0,0,0), default_solver=wgs, path:str='../BEMMedia/', reflector_path:str = './flat-lam2.stl', height:float = 0.12) -> None:
        '''
        AcousTools Context Objects  for 1 board above a reflector\n
        :param origin: The position of the origin for the setup \n
        :param default_solver: The solver to use when creating a focus, default wgs
        :param path: The path where BEM Medis is stored, this folder should have a subfolder called BEMCache
        :param reflector_path: The Path to the reflector to use
        :param height: The seperation to use

        Should be used in a `with` block eg
        ```
        from acoustools.HighLevel import TopWithReflectorContext

        with TopWithReflectorContext() as ctx:
            tb.create_focus()
            tb.send_solution()
        ```
        '''
        super().__init__(origin)

        self.solver = default_solver
        self.board: Tensor = transducers(16, z=height/2)

        self.path: str = path
        self.reflector: Mesh = load_scatterer(path + reflector_path)
        translate(self.reflector, dz=-1*height/2)


        self.H: Tensor = get_cache_or_compute_H(self.reflector, self.board, path=path)

        
    def create_focus(self, location:None|Tensor|Iterable = None) -> None:
        '''
        Create a focus at `location` using the default solver defined in class creation and stores the result in the object. \n
        The user does not need to know how the focus is created or what the ouput of the computation is \n
        Should be called before `send_solution`  or `visualise`
        :param location: The location of the focus
        '''
        if location is None:
            location: Tensor = self.origin
        
        if type(location) is not Tensor:
            location: Tensor = create_points(1,1,location[0], location[1], location[2])


        E: Tensor = compute_E(self.reflector, location, self.board, path=self.path, H=self.H)
        self.solution = self.solver(location, board = self.board, A=E)

    
    def visualise(self, size:float=0.03, origin:Tensor=None, plane:str='xz') -> None:
        '''
        Visualises the sound field that has been previously computed, see `acoustools.Visualise.Visualiser` for parameter details \n
        :param size: Size of the visualisation window from the origin (window will be 2*size)
        :param origin: Central point of the visualisation
        :param plane: Plane to visulaise in
        '''
        abc: tuple[Tensor] = ABC(size, origin=origin, plane=plane)
        Visualise(*abc, self.solution, colour_functions=[propagate_BEM_pressure], colour_function_args=[{'path':self.path, "H":self.H, 'board':self.board, 'scatterer':self.reflector}])

    def send_solution(self, ids=(1000)) -> None:
        '''
        Sends the solution to a connected physical device
        :param ids: The ids for the device connected
        '''
        if self.lev is None: self.lev = LevitatorController(ids=ids)

        self.lev.levitate(self.solution)



class BottomWithReflectorContext(TopWithReflectorContext):

    def __init__(self, origin:Iterable|Tensor=(0,0,0), default_solver=wgs, path='../BEMMedia/', reflector_path = './flat-lam2.stl', height = 0.12) -> None:
        '''
        AcousTools Context Objects  for 1 board below a reflector\n
        :param origin: The position of the origin for the setup \n
        :param default_solver: The solver to use when creating a focus, default wgs
        :param path: The path where BEM Medis is stored, this folder should have a subfolder called BEMCache
        :param reflector_path: The Path to the reflector to use
        :param height: The seperation to use

        Should be used in a `with` block eg
        ```
        from acoustools.HighLevel import TopWithReflectorContext

        with TopWithReflectorContext() as ctx:
            tb.create_focus()
            tb.send_solution()
        ```
        '''
        super().__init__(origin, default_solver, path, reflector_path, height)

        self.board: Tensor = transducers(16, z=-1*height/2)
        
        self.reflector: Mesh = load_scatterer(path + reflector_path)
        translate(self.reflector, dz=height/2)


        self.H: Tensor = get_cache_or_compute_H(self.reflector, self.board, path=path)