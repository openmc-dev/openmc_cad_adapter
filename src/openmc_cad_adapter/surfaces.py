from abc import ABC, abstractmethod
import sys
import math
import warnings

import numpy as np
import openmc

from .cubit_util import emit_get_last_id, lastid
from .geom_util import move, rotate

def indent(indent_size):
    return ' ' * (2*indent_size)


class CADSurface(ABC):

    def to_cubit_surface(self, ent_type, node, extents, inner_world=None, hex=False):
        ids, cmds = self.to_cubit_surface_inner(ent_type, node, extents, inner_world, hex)
        # TODO: Add boundary condition to the correct surface(s)
        # cmds += self.boundary_condition(ids)
        return ids, cmds

    @abstractmethod
    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False):
        raise NotImplementedError

    def boundary_condition(self, cad_surface_ids):
        if self.boundary_type == 'transmission':
            return []
        cmds = []
        cmds.append(f'group \"boundary:{self.boundary_type}\" add surface {cad_surface_ids[2:]}')
        return cmds

    @classmethod
    def from_openmc_surface(cls, surface):
        with warnings.catch_warnings() as w:
            warnings.simplefilter("ignore")
            return cls.from_openmc_surface_inner(surface)

    @classmethod
    @abstractmethod
    def from_openmc_surface_inner(cls, surface):
        raise NotImplementedError


class CADPlane(CADSurface, openmc.Plane):

    @staticmethod
    def lreverse(node):
        return "" if node.side == '-' else "reverse"

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False):
        cmds = []

        n = np.array([self.coefficients[k] for k in ('a', 'b', 'c')])
        cd = self.coefficients['d']
        maxv = sys.float_info.min
        for i, v in enumerate(n):
            if abs(v) > maxv:
                maxv = v

        ns = cd * n

        cmds.append( f"create surface rectangle width  { 2*extents[0] } zplane")
        sur, cmd = emit_get_last_id( "surface" )
        cmds.append(cmd)
        surf, cmd = emit_get_last_id( "body" )

        n = n/np.linalg.norm(n)
        ns = cd * n
        zn = np.array([ 0, 0, 1 ])
        n3 = np.cross(n, zn)
        dot = np.dot(n, zn)
        cmds.append(f"# n3 {n3[0]} {n3[1]} {n3[2]}")
        degs = math.degrees(math.acos(np.dot(n, zn)))
        y = np.linalg.norm(n3)
        x = dot
        angle = - math.degrees(math.atan2( y, x ))
        if n3[0] != 0 or n3[1] != 0 or n3[2] != 0:
            cmds.append(f"Rotate body {{ {surf} }} about 0 0 0 direction {n3[0]} {n3[1]} {n3[2]} Angle {angle}")
        cmds.append(f"body {{ { surf } }} move {ns[0]} {ns[1]} {ns[2]}")
        cmds.append(f"brick x {extents[0]} y {extents[1]} z {extents[2]}" )
        ids = emit_get_last_id( ent_type )
        cmds.append(f"section body {{ {ids} }} with surface {{ {sur} }} {self.lreverse(node)}")
        cmds.append(f"del surface {{ {sur} }}")

        cmds += self.boundary_condition(ids)
        return ids, cmds

    @classmethod
    def from_openmc_surface_inner(cls, plane):
        return cls(plane.a, plane.b, plane.c, plane.d, plane.boundary_type, plane.albedo, plane.name, plane.id)


class CADXPlane(CADSurface, openmc.XPlane):

    @staticmethod
    def reverse(node):
        return "reverse" if node.side == '-' else ""

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False):
        cad_cmds = []
        cad_cmds.append(f"brick x {extents[0]} y {extents[1]} z {extents[2]}")
        ids = emit_get_last_id( ent_type, cad_cmds)
        cad_cmds.append(f"section body {{ {ids} }} with xplane offset {self.coefficients['x0']} {self.reverse(node)}")
        return ids, cad_cmds

    @classmethod
    def from_openmc_surface_inner(cls, plane):
        return cls(x0=plane.x0, boundary_type=plane.boundary_type, albedo=plane.albedo, name=plane.name, surface_id=plane.id)


class CADYPlane(CADSurface, openmc.YPlane):

    @staticmethod
    def reverse(node):
        return "reverse" if node.side == '-' else ""

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False):
        cad_cmds = []
        cad_cmds.append(f"brick x {extents[0]} y {extents[1]} z {extents[2]}")
        ids = emit_get_last_id( ent_type, cad_cmds)
        cad_cmds.append(f"section body {{ {ids} }} with yplane offset {self.coefficients['y0']} {self.reverse(node)}")
        return ids, cad_cmds

    @classmethod
    def from_openmc_surface_inner(cls, plane):
        return cls(y0=plane.y0, boundary_type=plane.boundary_type, albedo=plane.albedo, name=plane.name, surface_id=plane.id)


class CADZPlane(CADSurface, openmc.ZPlane):

    @staticmethod
    def reverse(node):
        return "reverse" if node.side == '-' else ""

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False):
        cad_cmds = []
        cad_cmds.append(f"brick x {extents[0]} y {extents[1]} z {extents[2]}")
        ids = emit_get_last_id( ent_type, cad_cmds)
        cad_cmds.append(f"section body {{ {ids} }} with zplane offset {self.coefficients['z0']} {self.reverse(node)}")
        return ids, cad_cmds

    @classmethod
    def from_openmc_surface_inner(cls, plane):
        return cls(z0=plane.z0, boundary_type=plane.boundary_type, albedo=plane.albedo, name=plane.name, surface_id=plane.id)

class CADCylinder(CADSurface, openmc.Cylinder):

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False):
        print('XCADCylinder to cubit surface')
        cad_cmds = []
        h = inner_world[2] if inner_world else extents[2]
        cad_cmds.append(f"cylinder height {h} radius {self.r}")
        ids = emit_get_last_id(cmds=cad_cmds)
        if node.side != '-':
            wid = 0
            if inner_world:
                if hex:
                    cad_cmds.append(f"create prism height {inner_world[2]} sides 6 radius { ( inner_world[0] / 2 ) }")
                    wid = emit_get_last_id(ent_type, cad_cmds)
                    cad_cmds.append(f"rotate body {{ {wid} }} about z angle 30")
                else:
                    cad_cmds.append(f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}")
                    wid = emit_get_last_id(ent_type, cad_cmds)
            else:
                cad_cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                wid = emit_get_last_id(ent_type, cad_cmds)
            cad_cmds.append( f"subtract body {{ { ids } }} from body {{ { wid } }}" )
            rotate( wid, self.dx, self.dy, self.dz, cad_cmds)
            move( wid, self.x0, self.y0, self.z0, cad_cmds)
            return wid, cad_cmds
        rotate( ids, self.dx, self.dy, self.dz, cad_cmds)
        move( ids, self.x0, self.y0, self.z0, cad_cmds)
        return ids, cad_cmds

    @classmethod
    def from_openmc_surface_inner(cls, cyl):
        return cls(r=cyl.r, x0=cyl.x0, y0=cyl.y0, z0=cyl.z0, dx=cyl.dx, dy=cyl.dy, dz=cyl.dz,
                   boundary_type=cyl.boundary_type, albedo=cyl.albedo, name=cyl.name, surface_id=cyl.id)

class CADXCylinder(CADSurface, openmc.XCylinder):

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False):
        cad_cmds = []
        h = inner_world[0] if inner_world else extents[0]
        cad_cmds.append( f"cylinder height {h} radius {self.r}")
        ids = emit_get_last_id( ent_type , cad_cmds)
        cad_cmds.append(f"rotate body {{ {ids} }} about y angle 90")
        if node.side != '-':
            wid = 0
            if inner_world:
                if hex:
                    cad_cmds.append(f"create prism height {inner_world[2]} sides 6 radius { ( inner_world[0] / 2 ) }")
                    wid = emit_get_last_id(ent_type, cad_cmds)
                    cad_cmds.append(f"rotate body {{ {wid} }} about z angle 30")
                    cad_cmds.append(f"rotate body {{ {wid} }} about y angle 90")
                else:
                    cad_cmds.append(f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}")
                    wid = emit_get_last_id(ent_type, cad_cmds)
            else:
                cad_cmds.append( f"brick x {extents[0]} y {extents[1]} z {extents[2]}" )
                wid = emit_get_last_id( ent_type , cad_cmds)
            cad_cmds.append(f"subtract body {{ { ids } }} from body {{ { wid } }}")
            move(wid, 0, self.y0, self.z0, cad_cmds)
            return wid, cad_cmds
        move(ids, 0, self.y0, self.z0, cad_cmds)
        return ids, cad_cmds

    @classmethod
    def from_openmc_surface_inner(cls, cyl):
        return cls(r=cyl.r, y0=cyl.y0, z0=cyl.z0, boundary_type=cyl.boundary_type, albedo=cyl.albedo, name=cyl.name, surface_id=cyl.id)


class CADYCylinder(CADSurface, openmc.YCylinder):

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False):
        cad_cmds = []
        h = inner_world[1] if inner_world else extents[1]
        cad_cmds.append( f"cylinder height {h} radius {self.r}")
        ids = emit_get_last_id( ent_type , cad_cmds)
        cad_cmds.append(f"rotate body {{ {ids} }} about x angle 90")
        if node.side != '-':
            wid = 0
            if inner_world:
                if hex:
                    cad_cmds.append(f"create prism height {inner_world[2]} sides 6 radius { ( inner_world[0] / 2 ) }")
                    wid = emit_get_last_id(ent_type, cad_cmds)
                    cad_cmds.append(f"rotate body {{ {wid} }} about z angle 30")
                    cad_cmds.append(f"rotate body {{ {wid} }} about x angle 90")
                else:
                    cad_cmds.append(f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}")
                    wid = emit_get_last_id(ent_type, cad_cmds)
            else:
                cad_cmds.append( f"brick x {extents[0]} y {extents[1]} z {extents[2]}" )
                wid = emit_get_last_id( ent_type , cad_cmds)
            cad_cmds.append(f"subtract body {{ { ids } }} from body {{ { wid } }}")
            move(wid, self.x0, 0, self.z0, cad_cmds)
            return wid, cad_cmds
        move(ids, self.x0, 0, self.z0, cad_cmds)
        return ids, cad_cmds

    @classmethod
    def from_openmc_surface_inner(cls, cyl):
        return cls(r=cyl.r, x0=cyl.x0, z0=cyl.z0, boundary_type=cyl.boundary_type, albedo=cyl.albedo, name=cyl.name, surface_id=cyl.id)


class CADZCylinder(CADSurface, openmc.ZCylinder):

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False):
        cad_cmds = []
        h = inner_world[2] if inner_world else extents[2]
        cad_cmds.append( f"cylinder height {h} radius {self.r}")
        ids = emit_get_last_id( ent_type , cad_cmds)
        if node.side != '-':
            wid = 0
            if inner_world:
                if hex:
                    cad_cmds.append(f"create prism height {inner_world[2]} sides 6 radius { ( inner_world[0] / 2 ) }")
                    wid = emit_get_last_id(ent_type, cad_cmds)
                    cad_cmds.append(f"rotate body {{ {wid} }} about z angle 30")
                else:
                    cad_cmds.append(f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}")
                    wid = emit_get_last_id(ent_type, cad_cmds)
            else:
                cad_cmds.append( f"brick x {extents[0]} y {extents[1]} z {extents[2]}" )
                wid = emit_get_last_id( ent_type , cad_cmds)
            cad_cmds.append(f"subtract body {{ { ids } }} from body {{ { wid } }}")
            move(wid, self.x0, self.y0, 0, cad_cmds)
            return wid, cad_cmds
        move(ids, self.x0, self.y0, 0, cad_cmds)
        return ids, cad_cmds

    @classmethod
    def from_openmc_surface_inner(cls, cyl):
        return cls(r=cyl.r, x0=cyl.x0, y0=cyl.y0, boundary_type=cyl.boundary_type, albedo=cyl.albedo, name=cyl.name, surface_id=cyl.id)


class CADSphere(CADSurface, openmc.Sphere):

    def to_cubit_surface_inner(self, ent_type, node, extents, inner_world=None, hex=False):
        cad_cmds = []
        cad_cmds.append( f"sphere redius {self.r}")
        ids = emit_get_last_id(ent_type, cad_cmds)
        move(ids, self.x0, self.y0, self.z0, cad_cmds)
        return ids, cad_cmds

    @classmethod
    def from_openmc_surface_inner(cls, sphere):
        return cls(r=sphere.r, x0=sphere.x0, y0=sphere.y0, z0=sphere.z0, boundary_type=sphere.boundary_type, albedo=sphere.albedo, name=sphere.name, surface_id=sphere.id)
