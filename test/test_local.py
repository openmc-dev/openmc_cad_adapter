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
