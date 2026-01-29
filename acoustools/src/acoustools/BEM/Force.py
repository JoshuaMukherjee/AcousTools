
from acoustools.BEM import BEM_forward_model_second_derivative_mixed, BEM_forward_model_second_derivative_unmixed, BEM_forward_model_grad, compute_E, get_cache_or_compute_H, get_cache_or_compute_H_gradients
from acoustools.Utilities import TRANSDUCERS, propagate
from acoustools.Force import force_mesh
from acoustools.Mesh import load_scatterer, get_centres_as_points, get_normals_as_points, get_areas, scale_to_diameter,\
    centre_scatterer, translate, merge_scatterers, get_centre_of_mass_as_points, get_volume
from acoustools.Gorkov import get_gorkov_constants


import acoustools.Constants as c

from torch import Tensor
import torch

from vedo import Mesh

def BEM_compute_force(activations:Tensor, points:Tensor,board:Tensor|None=None,return_components:bool=False, V:float=c.V, scatterer:Mesh=None, 
                  H:Tensor=None, path:str="Media", k=c.k, transducer_radius=c.radius, p_ref=c.P_ref,
                  medium_density=c.p_0, medium_speed = c.c_0, particle_density = c.p_p, particle_speed = c.c_p) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    '''
    Returns the force on a particle using the analytical derivative of the Gor'kov potential and BEM\n
    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param board: Transducers to use, if `None` uses `acoustools.Utilities.TRANSDUCERS`
    :param return_components: If true returns force as one tensor otherwise returns Fx, Fy, Fz
    :param V: Particle volume
    :param scatterer: Scatterer to use
    :param H: H to use, will load/compute if None
    :param path: Path to folder containing BEMCache
    :return: force  
    '''

    #Bk.2 Pg.319

    if board is None:
        board = TRANSDUCERS
    
    F = compute_E(scatterer=scatterer,points=points,board=board,H=H, path=path, k=k, transducer_radius=transducer_radius, p_ref=p_ref)
    Fx, Fy, Fz = BEM_forward_model_grad(points,transducers=board,scatterer=scatterer,H=H, path=path, k=k, transducer_radius=transducer_radius, p_ref=p_ref)
    Fxx, Fyy, Fzz = BEM_forward_model_second_derivative_unmixed(points,transducers=board,scatterer=scatterer,H=H, path=path, k=k, transducer_radius=transducer_radius, p_ref=p_ref)
    Fxy, Fxz, Fyz = BEM_forward_model_second_derivative_mixed(points,transducers=board,scatterer=scatterer,H=H, path=path, k=k, transducer_radius=transducer_radius, p_ref=p_ref)

    p   = (F@activations)
    Px  = (Fx@activations)
    Py  = (Fy@activations)
    Pz  = (Fz@activations)
    Pxx = (Fxx@activations)
    Pyy = (Fyy@activations)
    Pzz = (Fzz@activations)
    Pxy = (Fxy@activations)
    Pxz = (Fxz@activations)
    Pyz = (Fyz@activations)


    grad_p = torch.stack([Px,Py,Pz])
    grad_px = torch.stack([Pxx,Pxy,Pxz])
    grad_py = torch.stack([Pxy,Pyy,Pyz])
    grad_pz = torch.stack([Pxz,Pyz,Pzz])


    p_term = p*grad_p.conj() + p.conj()*grad_p

    px_term = Px*grad_px.conj() + Px.conj()*grad_px
    py_term = Py*grad_py.conj() + Py.conj()*grad_py
    pz_term = Pz*grad_pz.conj() + Pz.conj()*grad_pz

    K1, K2 = get_gorkov_constants(V=V, c_0=medium_speed, c_p=particle_speed, p_0=medium_density, p_p=particle_density)
    
    grad_U = K1 * p_term - K2 * (px_term + py_term + pz_term)
    force = -1*(grad_U).squeeze().real

    if return_components:
        return force[0], force[1], force[2] 
    else:
        return force 

def torque_mesh_surface(activations:Tensor, scatterer:Mesh=None, board:Tensor|None=None, sum_elements = True, use_pressure:bool=False,
                       H:Tensor=None, diameter=c.wavelength*2, p_ref = c.P_ref,
                       path:str="Media", surface_path:str = "/Sphere-solidworks-lam2.stl",
                       surface:Mesh|None=None, use_cache_H:bool=True, 
                       E:Tensor|None=None, Ex:Tensor|None=None, Ey:Tensor|None=None, Ez:Tensor|None=None,
                        internal_points=None ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    '''
    Computes the force on a scattering obejct by computing thr force on a far field surface\\
    :param activations: Hologram
    :param scatterer: Object to compute force on
    :param board: Transducers
    :param sum_elements: if True will call sum across mesh elements
    :param H: H matrix to use for BEM, if None will be computed
    :param diameter: diameter of surfac to use
    :param path: path to BEMCache
    :param surface_path: Name of stl to use for surface such that path + surface path is the full address
    :param surface: Surface to use, if None will be laoded from surface_path
    :param use_cache_H: If true use BEM cache system
    :param E: E matrix to use for BEM, if None will be computed
    :param Ex: Grad E wrt x matrix to use for BEM, if None will be computed
    :param Ey: Grad E wrt y matrix to use for BEM, if None will be computed
    :param Ey: Grad E wrt z matrix to use for BEM, if None will be computed
    :returns torque:
    '''

    object_com = get_centre_of_mass_as_points(scatterer)

    if surface is None:
        surface = load_scatterer(path+surface_path)
        scale_to_diameter(surface,diameter, reset=False, origin=True)
        centre_scatterer(surface)
        translate(surface, dx = object_com[:,0].item(), dy=object_com[:,1].item(), dz = object_com[:,2].item())
    


    points = get_centres_as_points(surface)
    norms = get_normals_as_points(surface)
    areas = get_areas(surface)
    surface_com = get_centre_of_mass_as_points(surface)
    r = points - surface_com
    # print('dot',torch.sum(r * norms, dim=1) / torch.norm(r,2, dim=1)) 
    
    if use_pressure:
        if E is None:
            E = compute_E(scatterer, points, board, use_cache_H=use_cache_H, path=path, H=H,internal_points=internal_points, p_ref=p_ref)
        p = propagate(activations,points,board,A=E, p_ref=p_ref)
        pressure_square = torch.abs(p)**2
        pressure_time_average = 1/2 * pressure_square

        r_cross_n = torch.cross(r,norms, dim=1)
        pressure_term = pressure_time_average * r_cross_n * areas
        if sum_elements: pressure_term = torch.sum(pressure_term, dim=2)
    else:
        pressure_term = 0

    if Ex is None or Ey is None or Ez is None:
        Ex, Ey, Ez = BEM_forward_model_grad(points, scatterer, board, use_cache_H=use_cache_H, H=H, path=path,internal_points=internal_points, p_ref=p_ref)
    
    px = (Ex@activations).squeeze(2).unsqueeze(0)
    py = (Ey@activations).squeeze(2).unsqueeze(0)
    pz = (Ez@activations).squeeze(2).unsqueeze(0)

    grad  = torch.cat((px,py,pz),dim=1)
    velocity = grad /( 1j * c.p_0 * c.angular_frequency)
    # return torch.sum(velocity,dim=2)
    # r_cross_outer_conj = torch.cross(r, outer_conj)

    # time_average_outer_v = 1/2 * torch.einsum('bin,bjn -> bijn', r_cross_v, velocity.conj().resolve_conj()) #Takes two (B,3,N) vectors and computes the outer product on them - i think...
    vconj_dot_n = torch.sum(velocity.conj() * norms, dim=1, keepdim=True)
    time_average_velocity = 1/2 *(torch.cross((vconj_dot_n * r), velocity, dim=1)).real * areas
    if sum_elements: time_average_velocity = torch.sum(time_average_velocity, dim=2)
    torque = - pressure_term - c.p_0 * time_average_velocity

    return torque



def force_mesh_surface(activations:Tensor, scatterer:Mesh=None, board:Tensor|None=None,
                       return_components:bool=False, sum_elements:bool = True, return_momentum:bool = False,
                       H:Tensor=None, diameter:float=c.wavelength*2,
                       path:str="Media", surface_path:str = "/Sphere-solidworks-lam2.stl",
                       surface:Mesh|None=None, use_cache_H:bool=True, 
                       E:Tensor|None=None, Ex:Tensor|None=None, Ey:Tensor|None=None, Ez:Tensor|None=None, 
                       use_momentum:float=True, p_ref:float=c.P_ref, internal_points=None, k=c.k, transducer_radius=c.radius) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    '''
    Computes the torque on a scattering obejct by computing thr force on a far field surface\\
    :param activations: Hologram
    :param scatterer: Object to compute force on
    :param board: Transducers
    :param return_components: if True will return Fx,Fy,Fz else returns force
    :param sum_elements: if True will call sum across mesh elements
    :param return_momentum: if True will return total force and the momentum flux term
    :param H: H matrix to use for BEM, if None will be computed
    :param diameter: diameter of surface to use
    :param path: path to BEMCache
    :param surface_path: Name of stl to use for surface such that path + surface path is the full address
    :param surface: Surface to use, if None will be laoded from surface_path
    :param use_cache_H: If true use BEM cache system
    :param E: E matrix to use for BEM, if None will be computed
    :param Ex: Grad E wrt x matrix to use for BEM, if None will be computed
    :param Ey: Grad E wrt y matrix to use for BEM, if None will be computed
    :param Ey: Grad E wrt z matrix to use for BEM, if None will be computed
    :use_momentum: If true will use momentum flux terms
    :returns force:
    '''
    
    if surface is None:
        surface = load_scatterer(path+surface_path)
        scale_to_diameter(surface,diameter, reset=False, origin=True)
        centre_scatterer(surface)
        object_com = get_centre_of_mass_as_points(scatterer)
        translate(surface, dx = object_com[:,0].item(), dy=object_com[:,1].item(), dz = object_com[:,2].item())
        

    points = get_centres_as_points(surface)
    norms = get_normals_as_points(surface)
    areas = get_areas(surface)
    
    if E is None:
        E,F,G,H = compute_E(scatterer, points, board,path=path, H=H, return_components=True, use_cache_H=use_cache_H, p_ref=p_ref,internal_points=internal_points, k=k, transducer_radius=transducer_radius)
    
    force, momentum = force_mesh(activations, points,norms,areas,board=board,F=E, use_momentum=use_momentum, p_ref=p_ref, k=k, transducer_radius=transducer_radius,
                    grad_function=BEM_forward_model_grad, grad_function_args={'scatterer':scatterer,
                                                                                'H':H,
                                                                                'path':path,
                                                                                "internal_points":internal_points,
                                                                                }, 
                                                                                return_components=True,
                                                                                Ax = Ex, Ay=Ey, Az=Ez)



    if sum_elements: force=torch.sum(force, dim=2)

    if return_components:
        if not return_momentum:
            return (force[:,0]), (force[:,1]), (force[:,2])
        else:
            return (force[:,0]), (force[:,1]), (force[:,2]), momentum
   
    if return_momentum: return force, momentum
    return force

def force_mesh_surface_divergance(activations:Tensor, scatterer:Mesh=None, board:Tensor|None=None,
                       sum_elements:bool = True, H:Tensor=None, diameter:float=c.wavelength*2,
                       path:str="Media", surface_path:str = "/Sphere-solidworks-lam2.stl",
                       surface:Mesh|None=None, use_cache_H:bool=True, force:Tensor|None=None, norms:Tensor|None=None, p_ref=c.P_ref, k=c.k, transducer_radius=c.radius) -> Tensor:
    '''
    Computes the divergance force (the dot product of the force and normals on the surface) on a scattering obejct by computing thr force on a far field surface\\
    :param activations: Hologram
    :param scatterer: Object to compute force on
    :param board: Transducers
    :param sum_elements: if True will call sum across mesh elements
    :param H: H matrix to use for BEM, if None will be computed
    :param diameter: diameter of surfac to use
    :param path: path to BEMCache
    :param surface_path: Name of stl to use for surface such that path + surface path is the full address
    :param surface: Surface to use, if None will be laoded from surface_path
    :param use_cache_H: If true use BEM cache system
    :param force: Precomputed force, if non is computed using `force_mesh_surface`
    :param norms: Precomputed norms, if non is found from surface
    :returns divergance of force:
    '''


    if surface is None:
        surface = load_scatterer(path+surface_path)
        scale_to_diameter(surface,diameter, reset=False, origin=True)
        centre_scatterer(surface)
        object_com = get_centre_of_mass_as_points(scatterer)
        translate(surface, dx = object_com[:,0].item(), dy=object_com[:,1].item(), dz = object_com[:,2].item())
    
    if force is None:
        force = force_mesh_surface(activations, scatterer, board, H=H, diameter=diameter, path=path, 
                               surface_path=surface_path, surface=surface, use_cache_H=use_cache_H, sum_elements=False, use_momentum=True, k=k, p_ref=p_ref, transducer_radius=transducer_radius) 

    if norms is None: norms = get_normals_as_points(surface)
    areas = get_areas(surface)

    div = (torch.sum(norms * force, dim=1) * areas )

    if sum_elements: div = torch.sum(div, dim=1)

    v = get_volume(surface)

    return div / v


def force_mesh_surface_curl(activations:Tensor, scatterer:Mesh=None, board:Tensor|None=None,
                       sum_elements:bool = True, H:Tensor=None, diameter:float=c.wavelength*2,
                       path:str="Media", surface_path:str = "/Sphere-solidworks-lam2.stl",
                       surface:Mesh|None=None, use_cache_H:bool=True, magnitude:Tensor|None = False, force:Tensor|None=None, p_ref=c.P_ref, k=c.k, transducer_radius=c.radius) -> Tensor:
    '''
    Computes the curl force (the cross product of the force and normals on the surface) on a scattering obejct by computing thr force on a far field surface\\
    :param activations: Hologram
    :param scatterer: Object to compute force on
    :param board: Transducers
    :param sum_elements: if True will call sum across mesh elements
    :param H: H matrix to use for BEM, if None will be computed
    :param diameter: diameter of surfac to use
    :param path: path to BEMCache
    :param surface_path: Name of stl to use for surface such that path + surface path is the full address
    :param surface: Surface to use, if None will be laoded from surface_path
    :param use_cache_H: If true use BEM cache system
    :param force: Precomputed force, if non is computed using `force_mesh_surface`
    :param magnitude: If true will call `torch.norm` on the curl vectors
    :returns curl of force:
    '''
    
    if force is None: force = force_mesh_surface(activations, scatterer, board, H=H, diameter=diameter, path=path, 
                               surface_path=surface_path, surface=surface, use_cache_H=use_cache_H, sum_elements=False, k=k, p_ref=p_ref, transducer_radius=transducer_radius) 

    if surface is None:
        surface = load_scatterer(path+surface_path)
        scale_to_diameter(surface,diameter, reset=False, origin=True)
        centre_scatterer(surface)
        object_com = get_centre_of_mass_as_points(scatterer)
        translate(surface, dx = object_com[:,0].item(), dy=object_com[:,1].item(), dz = object_com[:,2].item())
        
    norms = get_normals_as_points(surface).real
    areas = get_areas(surface)

    curl = torch.cross(force, norms, dim=1) * areas 

    if sum_elements: curl = torch.sum(curl, dim=2)

    v = get_volume(surface)

    curl = curl/v

    if magnitude: return torch.norm(curl, dim=1)

    return curl 


def get_force_mesh_along_axis(start:Tensor,end:Tensor, activations:Tensor, scatterers:list[Mesh], board:Tensor, mask:Tensor|None=None, steps:int=200, 
                              path:str="Media",print_lines:bool=False, use_cache:bool=True, 
                              Hs:Tensor|None = None, Hxs:Tensor|None=None, Hys:Tensor|None=None, Hzs:Tensor|None=None, p_ref=c.P_ref, k=c.k, transducer_radius=c.radius) -> tuple[list[Tensor],list[Tensor],list[Tensor]]:
    '''
    @private
    Computes the force on a mesh at each point from `start` to `end` with number of samples = `steps`  \n
    :param start: The starting position
    :param end: The ending position
    :param activations: Transducer hologram
    :param scatterers: First element is the mesh to move, rest is considered static reflectors 
    :param board: Transducers to use 
    :param mask: The mask to apply to filter force for only the mesh to move
    :param steps: Number of steps to take from start to end
    :param path: path to folder containing BEMCache/ 
    :param print_lines: if true prints messages detaling progress
    :param use_cache: If true uses the cache system, otherwise computes H and does not save it
    :param Hs: List of precomputed forward propagation matricies
    :param Hxs: List of precomputed derivative of forward propagation matricies wrt x
    :param Hys: List of precomputed derivative of forward propagation matricies wrt y
    :param Hzs: List of precomputed derivative of forward propagation matricies wrt z
    :return: list for each axis of the force at each position
    '''

    print("get_force_mesh_along_axis - implementation H grad incorrect - do not use")
    # if Ax is None or Ay is None or Az is None:
    #     Ax, Ay, Az = grad_function(points=points, transducers=board, **grad_function_args)
    direction = (end - start) / steps

    translate(scatterers[0], start[0].item() - direction[0].item(), start[1].item() - direction[1].item(), start[2].item() - direction[2].item())
    scatterer = merge_scatterers(*scatterers)

    points = get_centres_as_points(scatterer)
    if mask is None:
        mask = torch.ones(points.shape[2]).to(bool)

    Fxs = []
    Fys = []
    Fzs = []

    for i in range(steps+1):
        if print_lines:
            print(i)
        
        
        translate(scatterers[0], direction[0].item(), direction[1].item(), direction[2].item())
        scatterer = merge_scatterers(*scatterers)

        points = get_centres_as_points(scatterer)
        areas = get_areas(scatterer)
        norms = get_normals_as_points(scatterer)

        if Hs is None:
            H = get_cache_or_compute_H(scatterer, board, path=path, print_lines=print_lines, use_cache_H=use_cache, k=k, p_ref=p_ref, transducer_radius=transducer_radius)
        else:
            H = Hs[i]
        
        if Hxs is None or Hys is None or Hzs is None:
            Hx, Hy, Hz = get_cache_or_compute_H_gradients(scatterer, board, path=path, print_lines=print_lines, use_cache_H_grad=use_cache)
        else:
            Hx = Hxs[i]
            Hy = Hys[i]
            Hz = Hzs[i]
        

        force = force_mesh(activations, points, norms, areas, board, F=H, Ax=Hx, Ay=Hy, Az=Hz)

        force = torch.sum(force[:,:,mask],dim=2).squeeze()
        Fxs.append(force[0])
        Fys.append(force[1])
        Fzs.append(force[2])
        
        # print(i, force[0].item(), force[1].item(),force[2].item())
    return Fxs, Fys, Fzs
