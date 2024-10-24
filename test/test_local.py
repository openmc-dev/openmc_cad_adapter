import openmc
import pytest

from openmc_cad_adapter import to_cubit_journal

from .test_utilities import diff_gold_file


# Test the XCylinder and YCylinder classes, the ZCylinder surface is tested
# extensively in the OpenMC example tests

def test_xcylinder(request):
    x_cyl = openmc.XCylinder(r=1.0, y0=10.0, z0=5.0)
    g = openmc.Geometry([openmc.Cell(region=-x_cyl)])
    to_cubit_journal(g, world=(500, 500, 500), filename='xcylinder.jou')
    diff_gold_file('xcylinder.jou')


def test_ycylinder(request):
    y_cyl = openmc.YCylinder(r=1.0, x0=10.0, z0=5.0)
    g = openmc.Geometry([openmc.Cell(region=-y_cyl)])
    to_cubit_journal(g, world=(500, 500, 500), filename='ycylinder.jou')
    diff_gold_file('ycylinder.jou')


def test_cylinder(request):
    cyl = openmc.Cylinder(x0=0.0, y0=0.0, z0=0.0, r=6.0, dx=0.7071, dy=0.7071, dz=0.0)
    g = openmc.Geometry([openmc.Cell(region=-cyl)])
    to_cubit_journal(g, world=(500, 500, 500), filename='cylinder.jou')
    diff_gold_file('cylinder.jou')


def test_x_cone(request):
    x_cone = openmc.XCone(x0=30.0, y0=3.0, z0=5.0, r2=5.0)
    g = openmc.Geometry([openmc.Cell(region=-x_cone)])
    to_cubit_journal(g, world=(500, 500, 500), filename='x_cone.jou')
    diff_gold_file('x_cone.jou')


def test_y_cone(request):
    y_cone = openmc.YCone(x0=40.0, y0=20.0, z0=7.0, r2=2.0)
    g = openmc.Geometry([openmc.Cell(region=-y_cone)])
    to_cubit_journal(g, world=(500, 500, 500), filename='y_cone.jou')
    diff_gold_file('y_cone.jou')


def test_z_cone(request):
    z_cone = openmc.ZCone(x0=50.0, y0=10.0, z0=2.0, r2=1.0)
    g = openmc.Geometry([openmc.Cell(region=-z_cone)])
    to_cubit_journal(g, world=(500, 500, 500), filename='z_cone.jou')
    diff_gold_file('z_cone.jou')