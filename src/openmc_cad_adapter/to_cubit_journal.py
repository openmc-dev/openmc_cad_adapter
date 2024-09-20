from argparse import ArgumentParser
from collections.abc import Iterable
import math
from numbers import Real
from pathlib import Path
import sys

from numpy.linalg import matrix_rank
import numpy as np

try:
    import openmc
except ImportError as e:
    raise type(e)("Please install OpenMC's Python API to use the CAD conversion tool")

from openmc.region import Region, Complement, Intersection, Union
from openmc.surface import Halfspace, Quadric
from openmc.lattice import Lattice, HexLattice

from .gqs import characterize_general_quadratic
from .cubit_util import emit_get_last_id, reset_cubit_ids, new_variable
from .geom_util import rotate, move

from .surfaces import CADPlane, CADXPlane, CADYPlane, CADZPlane

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


def to_cubit_journal(geometry : openmc.Geometry, world : Iterable[Real] = None,
                     cells: Iterable[int, openmc.Cell] = None,
                     filename: str = "openmc.jou",
                     to_cubit: bool = False,
                     seen: set = set()):
    """Convert an OpenMC geometry to a Cubit journal.

    Parameters
    ----------
        geometry : openmc.Geometry
            The geometry to convert to a Cubit journal.
        world : Iterable[Real], optional
            Extents of the model in X, Y, and Z. Defaults to None.
        cells : Iterable[int, openmc.Cell], optional
            List of cells or cell IDs to write to individual journal files. If None,
            all cells will be written to the same journal file. Defaults to None.
        filename : str, optional
            Output filename. Defaults to "openmc.jou".
        to_cubit : bool, optional
            Uses the cubit Python module to write the model as a .cub5 file.
            Defaults to False.
        seen : set, optional
            Internal parameter.

    """
    reset_cubit_ids()

    if not filename.endswith('.jou'):
        filename += '.jou'

    if isinstance(geometry, openmc.Model):
        geometry = geometry.geometry

    if cells is not None:
        cells = [c if not isinstance(c, openmc.Cell) else c.id for c in cells]

    if to_cubit:
        try:
            import cubit
        except ImportError:
            raise ImportError("Cubit Python API not found. Please install Cubit to use this feature.")

    geom = geometry

    if world is None:
        bbox = geometry.bounding_box
        if not all(np.isfinite(bbox[0])) or not all(np.isfinite(bbox[1])):
            raise RuntimeError('Model bounds were not provided and the bounding box determined by OpenMC is not finite.'
                               'Please provide a world size argument to proceed')
        # to ensure that the box
        box_max = np.max(np.abs(bbox[0], bbox[1]).T)
        world_size = (2 * box_max, 2 * box_max, 2 * box_max)

    if world is None:
        raise RuntimeError("Model extents could not be determined automatically and must be provided manually")

    w = world

    cmds = []
    cmds.extend( [
        "set graphics off",
        "set journal off",
        #"set undo off",
        #"set default autosizing off",   # especially when the CAD is complex (contains many spline surfaces) this can have a massive effect
        #"set info off",
        #"set warning off",
        ])
    def cubit_cmd(s):
        cmds.append(s)


    def surface_to_cubit_journal(node, w, indent = 0, inner_world = None, hex = False, ent_type = "body" ):
        def ind():
            return ' ' * (2*indent)
        if isinstance(node, Halfspace):
                seen.add( node.surface )
                surface = node.surface

                nonlocal cmds

                def reverse():
                    return "reverse" if node.side == '-' else ""

                if surface._type == "plane":
                    CADSurface = CADPlane.from_openmc_plane(surface)
                    ids, cad_cmds = CADSurface.to_cubit_surface(ent_type, node, w)
                    cmds += cad_cmds
                    return ids
                elif surface._type == "x-plane":
                    cad_surface = CADXPlane.from_openmc_surface(surface)
                    ids, surface_cmds = cad_surface.to_cubit_surface(ent_type, node, w)
                    cmds += surface_cmds
                    return ids
                elif surface._type == "y-plane":
                    cad_surface = CADYPlane.from_openmc_surface(surface)
                    ids, surface_cmds = cad_surface.to_cubit_surface(ent_type, node, w)
                    cmds += surface_cmds
                    return ids
                elif surface._type == "z-plane":
                    cad_surface = CADZPlane.from_openmc_surface(surface)
                    ids, surface_cmds = cad_surface.to_cubit_surface(ent_type, node, w)
                    cmds += surface_cmds
                    return ids
                elif surface._type == "cylinder":
                    h = inner_world[2] if inner_world else w[2]
                    cmds.append(f"cylinder height {h} radius {surface.coefficients['r']}")
                    ids = emit_get_last_id(cmds=cmds)
                    if node.side != '-':
                        wid = 0
                        if inner_world:
                            if hex:
                                cmds.append(f"create prism height {inner_world[2]} sides 6 radius { ( inner_world[0] / 2 ) }")
                                wid = emit_get_last_id(ent_type, cmds)
                                cmds.append(f"rotate body {{ {wid} }} about z angle 30")
                            else:
                                cmds.append(f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}")
                                wid = emit_get_last_id(ent_type, cmds)
                        else:
                            cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                            wid = emit_get_last_id(ent_type, cmds)
                        cmds.append( f"subtract body {{ { ids } }} from body {{ { wid } }}" )
                        rotate( wid, surface.coefficients['dx'], surface.coefficients['dy'], surface.coefficients['dz'], cmds)
                        move( wid, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'], cmds)
                        return wid
                    rotate( ids, surface.coefficients['dx'], surface.coefficients['dy'], surface.coefficients['dz'], cmds)
                    move( ids, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'], cmds)
                    return ids
                elif surface._type == "x-cylinder":
                    h = inner_world[0] if inner_world else w[0]
                    cmds.append(f"cylinder height {h} radius {surface.coefficients['r']}")
                    ids = emit_get_last_id(ent_type , cmds)
                    cmds.append( f"rotate body {{ { ids } }} about y angle 90")
                    if node.side != '-':
                        wid = 0
                        if inner_world:
                            if hex:
                                cmds.append(f"create prism height {inner_world[2]} sides 6 radius { inner_world[0] / 2 } ")
                                wid = emit_get_last_id( ent_type , cmds)
                                cmds.append(f"rotate body {{ {wid} }} about z angle 30")
                                cmds.append(f"rotate body {{ {wid} }} about y angle 90")
                            else:
                                cmds.append(f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}")
                                wid = emit_get_last_id(ent_type, cmds)
                        else:
                            cmds.append(f"brick x {w[0]} y {w[1]} z {w[2]}")
                            wid = emit_get_last_id(ent_type, cmds)
                        cmds.append(f"subtract body {{ { ids } }} from body {{ { wid } }}")
                        move(wid, 0, surface.coefficients['y0'], surface.coefficients['z0'], cmds)
                        return wid
                    move(ids, 0, surface.coefficients['y0'], surface.coefficients['z0'], cmds)
                    return ids
                elif surface._type == "y-cylinder":
                    h = inner_world[1] if inner_world else w[1]
                    cmds.append(f"cylinder height {h} radius {surface.coefficients['r']}")
                    ids = emit_get_last_id(ent_type, cmds)
                    cmds.append(f"rotate body {{ {ids} }} about x angle 90")
                    if node.side != '-':
                        wid = 0
                        if inner_world:
                            if hex:
                                cmds.append(f"create prism height {inner_world[2]} sides 6 radius { ( inner_world[0] / 2) }")
                                wid = emit_get_last_id( ent_type , cmds)
                                cmds.append(f"rotate body {{ {wid} }} about z angle 30")
                                cmds.append(f"rotate body {{ {wid} }} about x angle 90")
                            else:
                                cmds.append(f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}")
                                wid = emit_get_last_id(ent_type, cmds)
                        else:
                            cmds.append(f"brick x {w[0]} y {w[1]} z {w[2]}")
                            wid = emit_get_last_id(ent_type, cmds)
                        cmds.append(f"subtract body {{ {ids} }} from body {{ {wid} }}")
                        move(wid, surface.coefficients['x0'], 0, surface.coefficients['z0'], cmds)
                        return wid
                    move(ids, surface.coefficients['x0'], 0, surface.coefficients['z0'], cmds)
                    return ids
                elif surface._type == "z-cylinder":
                    h = inner_world[2] if inner_world else w[2]
                    cmds.append( f"cylinder height {h} radius {surface.coefficients['r']}")
                    ids = emit_get_last_id( ent_type , cmds)
                    if node.side != '-':
                        wid = 0
                        if inner_world:
                            if hex:
                                cmds.append(f"create prism height {inner_world[2]} sides 6 radius { ( inner_world[0] / 2 ) }")
                                wid = emit_get_last_id(ent_type, cmds)
                                cmds.append(f"rotate body {{ {wid} }} about z angle 30")
                            else:
                                cmds.append(f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}")
                                wid = emit_get_last_id(ent_type, cmds)
                        else:
                            cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                            wid = emit_get_last_id( ent_type , cmds)
                        cmds.append(f"subtract body {{ { ids } }} from body {{ { wid } }}")
                        move(wid, surface.coefficients['x0'], surface.coefficients['y0'], 0, cmds)
                        return wid
                    move(ids, surface.coefficients['x0'], surface.coefficients['y0'], 0, cmds)
                    return ids
                elif surface._type == "sphere":
                    cmds.append( f"sphere redius {surface.coefficients['r']}")
                    ids = emit_get_last_id(ent_type, cmds)
                    move(ids, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'], cmds)
                    pass
                elif surface._type == "cone":
                    raise NotImplementedError("cone not implemented")
                    pass
                elif surface._type == "x-cone":
                    cmds.append( f"create frustum height {w[0]} radius {math.sqrt(surface.coefficients['r2']*w[0])} top 0")
                    ids = emit_get_last_id( ent_type , cmds)
                    cmds.append( f"rotate body {{ {ids} }} about y angle 90")
                    if node.side != '-':
                        cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                        wid = emit_get_last_id( ent_type , cmds)
                        cmds.append(f"subtract body {{ {ids} }} from body {{ {wid} }}")
                        move(wid, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'], cmds)
                        return wid
                    move(ids, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'], cmds)
                    return ids
                elif surface._type == "y-cone":
                    cmds.append( f"create frustum height {w[1]} radius {math.sqrt(surface.coefficients['r2']*w[1])} top 0")
                    ids = emit_get_last_id( ent_type , cmds)
                    cmds.append( f"rotate body {{ {ids} }} about x angle 90")
                    if node.side != '-':
                        cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                        wid = emit_get_last_id( ent_type , cmds)
                        cmds.append( f"subtract body {{ {ids} }} from body {{ {wid} }}" )
                        move( wid, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'], cmds )
                        return wid
                    move( ids, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'], cmds )
                    return ids
                elif surface._type == "z-cone":
                    cmds.append( f"create frustum height {w[2]} radius {math.sqrt(surface.coefficients['r2']*w[2])} top 0")
                    ids = emit_get_last_id( ent_type , cmds)
                    if node.side != '-':
                        cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                        wid = emit_get_last_id( ent_type , cmds)
                        cmds.append( f"subtract body {{ {ids} }} from body {{ {wid} }}" )
                        move( wid, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'], cmds )
                        return wid
                    move( ids, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'], cmds )
                    return ids
                elif surface._type == "x-torus":
                    cmds.append( f"torus major radius {surface.coefficients['a']} minor radius {surface.coefficients['b']}")
                    ids = emit_get_last_id( ent_type , cmds)
                    cmds.append( f"rotate body {{ {ids} }} about y angle 90")
                    if node.side != '-':
                        cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                        wid = emit_get_last_id( ent_type , cmds)
                        cmds.append( f"subtract body {{ {ids} }} from body {{ {wid} }}" )
                        move( wid, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'], cmds)
                        return wid
                    move( ids, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'], cmds)
                    return ids
                elif surface._type == "y-torus":
                    cmds.append( f"torus major radius {surface.coefficients['a']} minor radius {surface.coefficients['b']}")
                    ids = emit_get_last_id( ent_type , cmds)
                    cmds.append( f"rotate body {{ {ids} }} about x angle 90")
                    if node.side != '-':
                        cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                        wid = emit_get_last_id( ent_type , cmds)
                        cmds.append( f"subtract body {{ {id} }} from body {{ {wid} }}" )
                        move( wid, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'], cmds)
                        return wid
                    return ids
                elif surface._type == "z-torus":
                    cmds.append( f"torus major radius {surface.coefficients['a']} minor radius {surface.coefficients['b']}")
                    ids = emit_get_last_id( ent_type , cmds)
                    if node.side != '-':
                        cmds.append(f"brick x {w[0]} y {w[1]} z {w[2]}")
                        wid = emit_get_last_id(ent_type, cmds)
                        cmds.append(f"subtract body {{ {ids} }} from body {{ {wid} }}")
                        move(wid, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'], cmds)
                        return wid
                    move(ids, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'], cmds)
                    return ids
                elif surface._type == "quadric":
                    (gq_type, A_, B_, C_, K_, translation, rotation_matrix) = characterize_general_quadratic(surface)

                    def rotation_to_axis_angle( mat ):
                        x = mat[2, 1]-mat[1, 2];
                        y = mat[0, 2]-mat[2, 0];
                        z = mat[1, 0]-mat[0, 1];
                        r = math.hypot( x, math.hypot( y,z ));
                        t = mat[0,0] + mat[1,1] + mat[2,2];
                        theta = math.atan2(r,t-1);

                        if abs(theta) <= np.finfo(np.float64).eps:
                            return ( np.array([ 0, 0, 0 ]), 0 )
                        elif abs( theta - math.pi ) <= np.finfo(np.float64).eps:
                          # theta is pi (180 degrees) or extremely close to it
                          # find the column of mat with the largest diagonal
                          col = 0;
                          if mat[1,1] > mat[col,col]: col = 1
                          if mat[2,2] > mat[col,col]: col = 2

                          axis = np.array([ 0, 0, 0 ])

                          axis[col] = math.sqrt( (mat[col,col]+1)/2 )
                          denom = 2*axis[col]
                          axis[(col+1)%3] = mat[col,(col+1)%3] / denom
                          axis[(col+2)%3] = mat[col,(col+2)%3] / denom
                          return ( axis, theta )
                        else:
                          axis = np.array([ x/r, y/r, z/r ])
                          return ( axis, theta )
                    (r_axis, r_theta ) = rotation_to_axis_angle( rotation_matrix )
                    #compensate for cubits insertion of a negative
                    r_degs = - math.degrees( r_theta )
                    print( r_axis, math.degrees( r_theta ), r_degs )
                    if gq_type == ELLIPSOID : #1
                            r1 = math.sqrt( abs( -K_/A_ ) )
                            r2 = math.sqrt( abs( -K_/B_ ) )
                            r3 = math.sqrt( abs( -K_/C_ ) )
                            cmds.append( f"sphere redius 1")
                            ids = emit_get_last_id( ent_type , cmds)
                            cmds.append( f"body {{ { ids } }} scale x { r1 } y { r2 } z { r3 }")
                            move( ids, translation[0,0], translation[1,0], translation[2,0], cmds)
                    elif gq_type == ELLIPTIC_CYLINDER : #7
                        if A_ == 0:
                            print( "X", gq_type, A_, B_, C_, K_, r_axis, r_degs )
                            h = inner_world[0] if inner_world else w[0]
                            r1 = math.sqrt( abs( K_/C_ ) )
                            r2 = math.sqrt( abs( K_/B_ ) )
                            cmds.append( f"cylinder height {h} Major Radius {r1} Minor Radius {r2}")
                            ids = emit_get_last_id( ent_type , cmds)
                            cmds.append( f"rotate body {{ { ids } }} about y angle 90")
                            if node.side != '-':
                                wid = 0
                                if inner_world:
                                    if hex:
                                        cmds.append( f"create prism height {inner_world[2]} sides 6 radius { inner_world[0] / 2 } " )
                                        wid = emit_get_last_id( ent_type , cmds)
                                        cmds.append( f"rotate body {{ {wid} }} about z angle 30" )
                                        cmds.append( f"rotate body {{ {wid} }} about y angle 90")
                                    else:
                                        cmds.append( f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}" )
                                        wid = emit_get_last_id( ent_type , cmds)
                                else:
                                    cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                                    wid = emit_get_last_id( ent_type , cmds)
                                cmds.append( f"subtract body {{ { ids } }} from body {{ { wid } }}" )
                                cmds.append( f"Rotate body {{ {wid } }} about 0 0 0 direction {r_axis[0]} {r_axis[1]} {r_axis[2]} Angle {r_degs}")
                                move( wid, translation[0,0], translation[1,0], translation[2,0], cmds)
                                return wid
                            cmds.append( f"Rotate body {{ {ids} }} about 0 0 0 direction {r_axis[0]} {r_axis[1]} {r_axis[2]} Angle {r_degs}")
                            move( ids, translation[0,0], translation[1,0], translation[2,0], cmds)
                            return ids
                        if B_ == 0:
                            print( "Y", gq_type, A_, B_, C_, K_ )
                            h = inner_world[1] if inner_world else w[1]
                            r1 = math.sqrt( abs( K_/A_ ) )
                            r2 = math.sqrt( abs( K_/C_ ) )
                            cmds.append( f"cylinder height {h} Major Radius {r1} Minor Radius {r2}")
                            ids = emit_get_last_id( ent_type , cmds)
                            cmds.append( f"rotate body {{ { ids } }} about x angle 90")
                            if node.side != '-':
                                wid = 0
                                if inner_world:
                                    if hex:
                                        cmds.append( f"create prism height {inner_world[2]} sides 6 radius { inner_world[0] / 2 } " )
                                        wid = emit_get_last_id( ent_type , cmds)
                                        cmds.append( f"rotate body {{ {wid} }} about z angle 30" )
                                        cmds.append( f"rotate body {{ {wid} }} about y angle 90")
                                    else:
                                        cmds.append( f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}" )
                                        wid = emit_get_last_id( ent_type , cmds)
                                else:
                                    cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                                    wid = emit_get_last_id( ent_type , cmds)
                                cmds.append( f"subtract body {{ { ids } }} from body {{ { wid } }}" )
                                cmds.append( f"Rotate body {{ {wid } }} about 0 0 0 direction {r_axis[0]} {r_axis[1]} {r_axis[2]} Angle {r_degs}")
                                move( wid, translation[0,0], translation[1,0], translation[2,0], cmds)
                                return wid
                            cmds.append( f"Rotate body {{ {ids} }} about 0 0 0 direction {r_axis[0]} {r_axis[1]} {r_axis[2]} Angle {r_degs}")
                            move( ids, translation[0,0], translation[1,0], translation[2,0], cmds)
                            return ids
                        if C_ == 0:
                            print( "Z", gq_type, A_, B_, C_, K_ )
                            h = inner_world[2] if inner_world else w[2]
                            r1 = math.sqrt( abs( K_/A_ ) )
                            r2 = math.sqrt( abs( K_/B_ ) )
                            cmds.append( f"cylinder height {h} Major Radius {r1} Minor Radius {r2}")
                            ids = emit_get_last_id( ent_type , cmds)
                            if node.side != '-':
                                wid = 0
                                if inner_world:
                                    if hex:
                                        cmds.append( f"create prism height {inner_world[2]} sides 6 radius { inner_world[0] / 2 } " )
                                        wid = emit_get_last_id( ent_type , cmds)
                                        cmds.append( f"rotate body {{ {wid} }} about z angle 30" )
                                        cmds.append( f"rotate body {{ {wid} }} about y angle 90")
                                    else:
                                        cmds.append( f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}" )
                                        wid = emit_get_last_id( ent_type , cmds)
                                else:
                                    cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                                    wid = emit_get_last_id( ent_type , cmds)
                                cmds.append( f"subtract body {{ { ids } }} from body {{ { wid } }}" )
                                cmds.append( f"Rotate body {{ {wid } }} about 0 0 0 direction {r_axis[0]} {r_axis[1]} {r_axis[2]} Angle {r_degs}")
                                move( wid, translation[0,0], translation[1,0], translation[2,0], cmds)
                                return wid
                            cmds.append( f"Rotate body {{ {ids} }} about 0 0 0 direction {r_axis[0]} {r_axis[1]} {r_axis[2]} Angle {r_degs}")
                            move( ids, translation[0,0], translation[1,0], translation[2,0], cmds)
                            return ids
                    elif gq_type == ELLIPTIC_CONE : #3
                        if A_ == 0:
                            h = inner_world[0] if inner_world else w[0]
                            minor = math.sqrt( abs( -A_/C_ ) )
                            major = math.sqrt( abs( -A_/B_ ) )
                            rot_angle = - 90
                            rot_axis = 1
                            cmds.append( f"create frustum height {h} Major Radius {major} Minor Radius {minor} top 0")
                            ids = emit_get_last_id( ent_type , cmds)
                            cmds.append( f"rotate body {{ { ids } }} about y angle -90")
                            cmds.append( f"copy body {{ { ids } }}")
                            mirror = emit_get_last_id( ent_type , cmds)
                            cmds.append( f"rotate body {{ { mirror } }} about 0 0 0 angle 180")
                            cmds.append( f"unit body {{ { ids } }} {{ { mirror } }}")
                            cmds.append( f"Rotate body {{ {ids} }} about 0 0 0 direction {r_axis[0]} {r_axis[1]} {r_axis[2]} Angle {r_degs}")
                            move( ids, translation[0,0], translation[1,0], translation[2,0], cmds)
                            return ids
                        if B_ == 0:
                            h = inner_world[1] if inner_world else w[1]
                            minor = math.sqrt( abs( -B_/A_ ) )
                            major = math.sqrt( abs( -B_/C_ ) )
                            rot_angle = 90
                            rot_axis = 0
                            cmds.append( f"create frustum height {h} Major Radius {major} Minor Radius {minor} top 0")
                            ids = emit_get_last_id( ent_type , cmds)
                            cmds.append( f"rotate body {{ { ids } }} about x angle 90")
                            cmds.append( f"copy body {{ { ids } }}")
                            mirror = emit_get_last_id( ent_type , cmds)
                            cmds.append( f"rotate body {{ { mirror } }} about 0 0 0 angle 180")
                            cmds.append( f"unit body {{ { ids } }} {{ { mirror } }}")
                            cmds.append( f"Rotate body {{ {ids} }} about 0 0 0 direction {r_axis[0]} {r_axis[1]} {r_axis[2]} Angle {r_degs}")
                            move( ids, translation[0,0], translation[1,0], translation[2,0], cmds)
                            return ids
                        if C_ == 0:
                            h = inner_world[2] if inner_world else w[2]
                            minor = math.sqrt( abs( -C_/A_ ) )
                            major = math.sqrt( abs( -C_/B_ ) )
                            rot_angle = 180
                            rot_axis = 0
                            cmds.append( f"create frustum height {h} Major Radius {major} Minor Radius {minor} top 0")
                            ids = emit_get_last_id( ent_type , cmds)
                            cmds.append( f"copy body {{ { ids } }}")
                            mirror = emit_get_last_id( ent_type , cmds)
                            cmds.append( f"rotate body {{ { mirror } }} about 0 0 0 angle 180")
                            cmds.append( f"unit body {{ { ids } }} {{ { mirror } }}")
                            cmds.append( f"Rotate body {{ {ids} }} about 0 0 0 direction {r_axis[0]} {r_axis[1]} {r_axis[2]} Angle {r_degs}")
                            move( ids, translation[0,0], translation[1,0], translation[2,0], cmds)
                            return ids
                    else:
                        raise NotImplementedError(f"{surface.type} not implemented")
                else:
                    raise NotImplementedError(f"{surface.type} not implemented")
        elif isinstance(node, Complement):
            print( "Complement:" )
            id = surface_to_cubit_journal(node.node, w, indent + 1, inner_world, ent_type = ent_type )
            cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
            wid = emit_get_last_id( ent_type , cmds)
            cmds.append( f"subtract body {{ {id} }} from body {{ {wid} }}" )
            return emit_get_last_id( ent_type , cmds)
        elif isinstance(node, Intersection):
            last = 0
            if len( node ) > 0:
                last = surface_to_cubit_journal( node[0], w, indent + 1, inner_world, ent_type = ent_type ,)
                for subnode in node[1:]:
                    s = surface_to_cubit_journal( subnode, w, indent + 1, inner_world, ent_type = ent_type ,)
                    before = emit_get_last_id(cmds=cmds)
                    cmds.append( f"intersect {ent_type} {{ {last} }} {{ {s} }}" )
                    after = emit_get_last_id(cmds=cmds)
                    last = new_variable()
                    cmds.append( f"#{{{last} = ( {before} == {after} ) ? {s} : {after}}}" )
                if inner_world:
                    cmds.append( f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}" )
                    iwid = emit_get_last_id( ent_type , cmds)
                    cmds.append( f"intersect {ent_type} {{ {last} }} {{ {iwid} }}" )
                    return emit_get_last_id( ent_type , cmds)
            return last
        elif isinstance(node, Union):
            if len( node ) > 0:
                local_ent_type = "body"
                first = surface_to_cubit_journal( node[0], w, indent + 1, inner_world, ent_type = local_ent_type )
                for subnode in node[1:]:
                    s = surface_to_cubit_journal( subnode, w, indent + 1, inner_world, ent_type = local_ent_type )
                    cmds.append( f"unite {local_ent_type} {{ {first} }} {{ {s} }}" )
                if inner_world:
                    cmds.append( f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}" )
                    iwid = emit_get_last_id( local_ent_type , cmds)
                    cmds.append( f"intersect {ent_type} {{ {last} }} {{ {iwid} }}" )
                    return first
            return first
        elif isinstance(node, Quadric):
            pass
        else:
            raise NotImplementedError(f"{node} not implemented")

    def process_node_or_fill( node, w, indent = 0, offset = [0, 0], inner_world = None, outer_ll = None, ent_type = "body", hex = False ):
        def ind():
            return ' ' * (2*indent)
        seen.add( node )
        results = []
        if hasattr( node, "region" ) and not ( hasattr( node, "fill" ) and isinstance(node.fill, Lattice) ):
            if node.region != None:
                id = surface_to_cubit_journal( node.region, w, indent, inner_world, hex = hex )
                results.append( id )
            elif hex:
                cmds.append( f"create prism height {inner_world[2]} sides 6 radius { ( inner_world[0] / 2) }" )
                wid = emit_get_last_id( ent_type , cmds)
                cmds.append( f"rotate body {{ {wid} }} about z angle 30" )
                results.append( wid )

        if hasattr( node, "fill" ) and isinstance(node.fill, Lattice):
            id = process_node_or_fill( node.fill, w, indent + 1, offset, inner_world )
            results.append( id )
        if hasattr( node, "universes" ):
            pitch = node._pitch
            if isinstance( node, HexLattice ) :
                three_d_hex_lattice = len( node._pitch ) > 1
                #three_d_hex_lattice = len( node.center ) > 2
                #3d hex lattice
                if three_d_hex_lattice:
                    center = [ node.center[0], node.center[1], node.center[1] ]
                    ii = 0
                    for uss in node.universes:
                        z = ii * pitch[1]
                        j = 0
                        for us in uss:
                            k = 0
                            ring_id = ( len( uss ) - j -1 )
                            #print( "RING_ID", ring_id )
                            def draw_hex_cell( n, cell, x, y ):
                                #print( i, j, k, len( node.universes ), len( uss), len( us ), x, y )
                                id = process_node_or_fill( cell, [ w[0], w[1], w[2] ], indent + 1, offset = [x,y,z], inner_world=[ pitch[0], pitch[0], pitch[1] ], outer_ll=outer_ll if outer_ll else [ node.center[0], node.center[1] ], hex = True )
                                ids = str( id )
                                if isinstance( id, list ):
                                    ids = ' '.join( map(str, id) )
                                else:
                                    pass
                                if ids != '':
                                    cmds.append( f"move body {{ {ids} }} midpoint location {x} {y} {z}" )
                            side_to_side_diameter =  pitch[0]/2 * math.sqrt( 3 )
                            center_to_mid_side_diameter = ( ( pitch[0] / 2 ) * math.sin( math.pi / 6 ) ) + pitch[0] / 2
                            if ring_id < 2:
                                for u in us:
                                    for n, cell in u._cells.items():
                                        #print( n, cell )
                                        theta = 2 * math.pi * -k / len( us ) + math.pi / 2
                                        r = ( len( uss ) - j -1 ) * side_to_side_diameter
                                        x = r * math.cos( theta )
                                        y = r * math.sin( theta )
                                        draw_hex_cell( n, cell, x, y )
                                    k = k + 1
                            else:
                                x = 0
                                x_pos = 0
                                y_pos = 0
                                r = ring_id
                                for i in range( r, 0, -1 ):
                                    x_pos = x * center_to_mid_side_diameter;
                                    y_pos = ring_id * side_to_side_diameter - ( x ) * 0.5 * side_to_side_diameter;
                                    for n, cell in us[k]._cells.items():
                                        draw_hex_cell( n, cell, x_pos, y_pos )
                                    #print( r, k, x, x_pos, y_pos )
                                    k = k + 1
                                    x = x + 1
                                y_pos = ring_id * side_to_side_diameter - ( x ) * 0.5 * side_to_side_diameter;
                                for i in range( r, 0, -1 ):
                                    x_pos = x * center_to_mid_side_diameter;
                                    for n, cell in us[k]._cells.items():
                                        draw_hex_cell( n, cell, x_pos, y_pos )
                                    #print( r, k, x, x_pos, y_pos )
                                    y_pos = y_pos - side_to_side_diameter;
                                    k = k + 1
                                for i in range( r, 0, -1 ):
                                    x_pos = x * center_to_mid_side_diameter;
                                    y_pos = - ring_id * side_to_side_diameter + ( x ) * 0.5 * side_to_side_diameter;
                                    for n, cell in us[k]._cells.items():
                                        draw_hex_cell( n, cell, x_pos, y_pos )
                                    #print( r, k, x, x_pos, y_pos )
                                    k = k + 1
                                    x = x - 1
                                for i in range( r, 0, -1 ):
                                    x_pos = x * center_to_mid_side_diameter;
                                    y_pos = - ring_id * side_to_side_diameter - ( x ) * 0.5 * side_to_side_diameter;
                                    for n, cell in us[k]._cells.items():
                                        draw_hex_cell( n, cell, x_pos, y_pos )
                                    #print( r, k, x, x_pos, y_pos )
                                    k = k + 1
                                    x = x - 1
                                y_pos = - ring_id * side_to_side_diameter - ( x ) * 0.5 * side_to_side_diameter;
                                for i in range( r, 0, -1 ):
                                    x_pos = x * center_to_mid_side_diameter;
                                    for n, cell in us[k]._cells.items():
                                        draw_hex_cell( n, cell, x_pos, y_pos )
                                    #print( r, k, x, x_pos, y_pos )
                                    y_pos = y_pos + side_to_side_diameter;
                                    k = k + 1
                                for i in range( r, 0, -1 ):
                                    x_pos = x * center_to_mid_side_diameter;
                                    y_pos = ring_id * side_to_side_diameter + ( x ) * 0.5 * side_to_side_diameter;
                                    for n, cell in us[k]._cells.items():
                                        draw_hex_cell( n, cell, x_pos, y_pos )
                                    #print( r, k, x, x_pos, y_pos )
                                    k = k + 1
                                    x = x + 1
                            j = j + 1
                        ii = ii + 1
                #2d hex lattice
                else:
                    center = [ node.center[0], node.center[1] ]
                    i = 0
                    for us in node.universes:
                        j = 0
                        for u in us:
                            for n, cell in u._cells.items():
                                    #print( n, cell )
                                    theta = 2 * math.pi * -j / len( us ) + math.pi / 2
                                    r = ( len( uss ) - i -1 ) * pitch[0]
                                    x = r * math.cos( theta )
                                    y = r * math.sin( theta )
                                    #print( n, i, j, k, len( node.universes ), len( uss), len( us ), x, y, theta )
                                    id = process_node_or_fill( cell, [ w[0], w[1], w[2] ], indent + 1, offset = [x,y], inner_world=[ pitch[0], pitch[0], pitch[1] ], outer_ll=outer_ll if outer_ll else [ node.center[0], node.center[1] ], hex = True )
                                    ids = str( id )
                                    if isinstance( id, list ):
                                        ids = ' '.join( map(str, id) )
                                        #results.extend( id )
                                    else:
                                        #results.append( id )
                                        pass
                                    if ids != '':
                                        cmds.append( f"move body {{ {ids} }} midpoint location {x} {y} {z}" )
                            j = j + 1
                        i = i + 1

            elif isinstance( node, Lattice ):
                ll = [ node.lower_left[0], node.lower_left[1] ]
                if outer_ll:
                    ll = outer_ll
                ll[0] = ll[0] + pitch[0] / 2
                ll[1] = ll[1] + pitch[1] / 2
                i = 0
                for us in node.universes:
                    j = 0
                    for u in us:
                        for n, cell in u._cells.items():
                            x = ll[0] + j * pitch[0] + offset[0]
                            y = ll[1] + i * pitch[1] + offset[1]
                            #print(  ind(), "UCell:", n, cell )
                            id = process_node_or_fill( cell, [ w[0], w[1], w[2] ], indent + 1, offset = [x,y], inner_world=[ pitch[0], pitch[1], w[2] ], outer_ll=outer_ll if outer_ll else [ node.lower_left[0], node.lower_left[1] ] )
                            ids = str( id )
                            if isinstance( id, list ):
                                ids = ' '.join( map(str, id) )
                                #results.extend( id )
                            else:
                                #results.append( id )
                                pass
                            if ids != '':
                                cmds.append( f"move body {{ {ids} }} midpoint location {x} {y} 0 except z" )
                        j = j + 1
                    i = i + 1
        #FIXME rotate and tranlate
        r = flatten( results )
        if len( r ) > 0:
            if node.name:
                cmds.append( f"body {{ {r[0]} }} name \"{node.name}\"" )
            else:
                cmds.append( f"body {{ {r[0]} }} name \"Cell_{node.id}\"" )
        return r

    def do_cell(cell, cell_ids: Iterable[int] = None):
        before = len( cmds )
        cmds.append( f"#CELL {cell.id}" )
        vol_or_body = process_node_or_fill( cell, w )
        if cell.fill_type == "material":
            cmds.append( f'group \"Material_{cell.fill.id}\" add body {{ { vol_or_body[0] } }} ' )
        after = len( cmds )

        if cell_ids is not None and cell.id in cell_ids:
            if filename.endswith(".jou"):
                cell_filename = filename[:-4] + f"_cell{cell.id}.jou"
            else:
                cell_filename = filename + f"_cell{cell.id}"
            with open(cell_filename, "w" ) as f:
                for x in cmds[before:after]:
                    f.write( x + "\n" )

    for cell in geom.root_universe._cells.values():
        if cells is not None and cell.id in cells:
            do_cell( cell, cell_ids=cells)
        else:
            do_cell( cell )

    if filename:
        with open( filename, "w" ) as f:
            for x in cmds:
                f.write( x + "\n" )

    if to_cubit:
        cubit.cmd( "reset" )
        for x in cmds:
            cubit.cmd( x )
            cubit.cmd(f"save as {filename[:-4]}.cub overwrite")


def openmc_to_cad():
    """Command-line interface for OpenMC to CAD model conversion"""
    parser = ArgumentParser()
    parser.add_argument('input', help='Path to a OpenMC model.xml file or path to a directory containing OpenMC XMLs')
    parser.add_argument('-o', '--output', help='Output filename', default='openmc.jou')
    parser.add_argument('-w', '--world-size', help='Maximum width of the geometry in X, Y, and Z', nargs=3, type=int)
    parser.add_argument('-c', '--cells', help='List of cell IDs to convert', nargs='+', type=int)
    parser.add_argument('--to-cubit', help='Run  Cubit', default=False, action='store_true')
    parser.add_argument('--cubit-path', help='Path to Cubit bin directory', default=None, type=str)
    args = parser.parse_args()

    model_path = Path(args.input)

    if model_path.is_dir():
        if not (model_path / 'settings.xml').exists():
            raise IOError(f'Unable to locate settings.xml in {model_path}')
        model = openmc.Model.from_xml(*[model_path / f for f in ('geometry.xml', 'materials.xml', 'settings.xml')])
    else:
        model = openmc.Model.from_model_xml(model_path)

    if args.cubit_path is not None:
        sys.path.append(args.cubit_path)

    to_cubit_journal(model.geometry, world=args.world_size, filename=args.output, cells=args.cells, to_cubit=args.to_cubit)
