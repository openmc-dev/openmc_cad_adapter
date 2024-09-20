from abc import ABC, abstractclassmethod, abstractmethod

import sys
import math

import numpy as np
import openmc

from .cubit_util import emit_get_last_id, lastid

def indent(indent_size):
    return ' ' * (2*indent_size)


class CADSurface(ABC):

    @abstractmethod
    def to_cubit_surface(self, ent_type, node, model_extent):
        raise NotImplementedError

    @abstractclassmethod
    def from_openmc_surface(cls, surface):
        raise NotImplementedError


class CADPlane(CADSurface, openmc.Plane):

    @staticmethod
    def lreverse(node):
        return "" if node.side == '-' else "reverse"

    def to_cubit_surface(self, ent_type, node, model_extent):
        cmds = []

        n = np.array([self.coefficients[k] for k in ('a', 'b', 'c')])
        cd = self.coefficients['d']
        maxv = sys.float_info.min
        for i, v in enumerate(n):
            if abs(v) > maxv:
                maxv = v

        ns = cd * n

        cmds.append( f"create surface rectangle width  { 2*model_extent[0] } zplane")
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
        cmds.append(f"brick x {model_extent[0]} y {model_extent[1]} z {model_extent[2]}" )
        ids = emit_get_last_id( ent_type )
        cmds.append(f"section body {{ {ids} }} with surface {{ {sur} }} {self.lreverse(node)}")
        cmds.append(f"del surface {{ {sur} }}")

        return cmds

    @classmethod
    def from_openmc_surface(cls, plane):
        return cls(plane.a, plane.b, plane.c, plane.d, plane.boundary_type, plane.albedo, plane.name, plane.id)


class CADXPlane(openmc.XPlane):

    @staticmethod
    def reverse(node):
        return "reverse" if node.side == '-' else ""

    def to_cubit_surface(self, ent_type, node, model_extent):
        cad_cmds = []
        cad_cmds.append(f"brick x {model_extent[0]} y {model_extent[1]} z {model_extent[2]}")
        ids = emit_get_last_id( ent_type, cad_cmds)
        cad_cmds.append(f"section body {{ {ids} }} with xplane offset {self.coefficients['x0']} {self.reverse(node)}")
        return ids, cad_cmds

    @classmethod
    def from_openmc_surface(cls, plane):
        return cls(x0=plane.x0, boundary_type=plane.boundary_type, albedo=plane.albedo, name=plane.name, surface_id=plane.id)


class CADYPlane(openmc.YPlane):

    @staticmethod
    def reverse(node):
        return "reverse" if node.side == '-' else ""

    def to_cubit_surface(self, ent_type, node, model_extent):
        cad_cmds = []
        cad_cmds.append(f"brick x {model_extent[0]} y {model_extent[1]} z {model_extent[2]}")
        ids = emit_get_last_id( ent_type, cad_cmds)
        cad_cmds.append(f"section body {{ {ids} }} with yplane offset {self.coefficients['y0']} {self.reverse(node)}")
        return ids, cad_cmds

    @classmethod
    def from_openmc_surface(cls, plane):
        return cls(y0=plane.y0, boundary_type=plane.boundary_type, albedo=plane.albedo, name=plane.name, surface_id=plane.id)


class CADZPlane(openmc.ZPlane):

    @staticmethod
    def reverse(node):
        return "reverse" if node.side == '-' else ""

    def to_cubit_surface(self, ent_type, node, model_extent):
        cad_cmds = []
        cad_cmds.append(f"brick x {model_extent[0]} y {model_extent[1]} z {model_extent[2]}")
        ids = emit_get_last_id( ent_type, cad_cmds)
        cad_cmds.append(f"section body {{ {ids} }} with zplane offset {self.coefficients['z0']} {self.reverse(node)}")
        return ids, cad_cmds

    @classmethod
    def from_openmc_surface(cls, plane):
        return cls(z0=plane.z0, boundary_type=plane.boundary_type, albedo=plane.albedo, name=plane.name, surface_id=plane.id)
