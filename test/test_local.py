from functools import wraps

import pytest

import openmc

from openmc_cad_adapter import to_cubit_journal

from .test_utilities import diff_gold_file
from test import run_in_tmpdir


def reset_openmc_ids(func):
    """
    Decorator to reset the auto-generated IDs in OpenMC before running a test
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        openmc.reset_auto_ids()
        func(*args, **kwargs)
    return wrapper


@reset_openmc_ids
def test_planes(request, run_in_tmpdir):
    plane1 = openmc.Plane(a=1.0, b=1.0, c=0.0, d=-5.0)
    plane2 = openmc.Plane(a=1.0, b=1.0, c=0.0, d=5.0)
    plane3 = openmc.Plane(a=0.0, b=1.0, c=1.0, d=-5.0)
    plane4 = openmc.Plane(a=0.0, b=1.0, c=1.0, d=5.0)
    plane5 = openmc.Plane(a=1.0, b=0.0, c=1.0, d=-5.0)
    plane6 = openmc.Plane(a=1.0, b=0.0, c=1.0, d=5.0)
    g = openmc.Geometry([openmc.Cell(region=+plane1 & -plane2 & +plane3 & -plane4 & +plane5 & -plane6)])
    to_cubit_journal(g, world=(500, 500, 500), filename='plane.jou')
    diff_gold_file('plane.jou')


@reset_openmc_ids
def test_nested_spheres(request, run_in_tmpdir):
    inner_sphere = openmc.Sphere(r=10.0)
    middle_sphere = openmc.Sphere(r=20.0)
    outer_sphere = openmc.Sphere(r=30.0)

    inner_cell = openmc.Cell(region=-inner_sphere)
    middle_cell = openmc.Cell(region=+inner_sphere & -middle_sphere)
    outer_cell = openmc.Cell(region=+middle_sphere & -outer_sphere)

    g = openmc.Geometry([outer_cell, middle_cell, inner_cell])
    to_cubit_journal(g, world=(500, 500, 500), filename='nested_spheres.jou')
    diff_gold_file('nested_spheres.jou')


# Test the XCylinder and YCylinder classes, the ZCylinder surface is tested
# extensively in the OpenMC example tests
@reset_openmc_ids
def test_xcylinder(request, run_in_tmpdir):
    x_cyl = openmc.XCylinder(r=1.0, y0=10.0, z0=5.0)
    g = openmc.Geometry([openmc.Cell(region=-x_cyl)])
    to_cubit_journal(g, world=(500, 500, 500), filename='xcylinder.jou')
    diff_gold_file('xcylinder.jou')


@reset_openmc_ids
def test_ycylinder(request, run_in_tmpdir):
    y_cyl = openmc.YCylinder(r=1.0, x0=10.0, z0=5.0)
    g = openmc.Geometry([openmc.Cell(region=-y_cyl)])
    to_cubit_journal(g, world=(500, 500, 500), filename='ycylinder.jou')
    diff_gold_file('ycylinder.jou')


@reset_openmc_ids
def test_cylinder(request, run_in_tmpdir):
    cyl = openmc.Cylinder(x0=0.0, y0=0.0, z0=0.0, r=6.0, dx=0.7071, dy=0.7071, dz=0.0)
    g = openmc.Geometry([openmc.Cell(region=-cyl)])
    to_cubit_journal(g, world=(500, 500, 500), filename='cylinder.jou')
    diff_gold_file('cylinder.jou')


@reset_openmc_ids
def test_x_cone(request, run_in_tmpdir):
    x_cone = openmc.XCone(x0=30.0, y0=3.0, z0=5.0, r2=5.0)
    g = openmc.Geometry([openmc.Cell(region=-x_cone)])
    to_cubit_journal(g, world=(500, 500, 500), filename='x_cone.jou')
    diff_gold_file('x_cone.jou')


@reset_openmc_ids
def test_y_cone(request, run_in_tmpdir):
    y_cone = openmc.YCone(x0=40.0, y0=20.0, z0=7.0, r2=2.0)
    g = openmc.Geometry([openmc.Cell(region=-y_cone)])
    to_cubit_journal(g, world=(500, 500, 500), filename='y_cone.jou')
    diff_gold_file('y_cone.jou')


@reset_openmc_ids
def test_z_cone(request, run_in_tmpdir):
    z_cone = openmc.ZCone(x0=50.0, y0=10.0, z0=2.0, r2=0.25)
    g = openmc.Geometry([openmc.Cell(region=-z_cone)])
    to_cubit_journal(g, world=(500, 500, 500), filename='z_cone.jou')
    diff_gold_file('z_cone.jou')


@reset_openmc_ids
def test_x_torus(request, run_in_tmpdir):
    x_torus = openmc.XTorus(x0=10.0, y0=10.0, z0=10.0, a=5.0, b=2.0, c=2.0)
    g = openmc.Geometry([openmc.Cell(region=-x_torus)])
    to_cubit_journal(g, world=(500, 500, 500), filename='x_torus.jou')
    diff_gold_file('x_torus.jou')


@reset_openmc_ids
def test_y_torus(request, run_in_tmpdir):
    y_torus = openmc.YTorus(x0=-10.0, y0=-10.0, z0=-10.0, a=5.0, b=2.0, c=2.0)
    g = openmc.Geometry([openmc.Cell(region=-y_torus)])
    to_cubit_journal(g, world=(500, 500, 500), filename='y_torus.jou')
    diff_gold_file('y_torus.jou')


@reset_openmc_ids
def test_z_torus(request, run_in_tmpdir):
    z_torus = openmc.ZTorus(x0=50.0, y0=50.0, z0=50.0, a=5.0, b=2.0, c=2.0)
    g = openmc.Geometry([openmc.Cell(region=-z_torus)])
    to_cubit_journal(g, world=(500, 500, 500), filename='z_torus.jou')
    diff_gold_file('z_torus.jou')


@reset_openmc_ids
def test_torus_diff_radii(request, run_in_tmpdir):
    with pytest.raises(ValueError):
        z_torus = openmc.ZTorus(x0=50.0, y0=50.0, z0=50.0, a=5.0, b=2.0, c=3.0)
        g = openmc.Geometry([openmc.Cell(region=-z_torus)])
        to_cubit_journal(g, world=(500, 500, 500), filename='a_torus.jou')


@reset_openmc_ids
def test_general_cone(request, run_in_tmpdir):
    with pytest.raises(NotImplementedError):
        cone = openmc.Cone(x0=0.0, y0=0.0, z0=0.0, r2=6.0, dx=1, dy=1, dz=1)
        g = openmc.Geometry([openmc.Cell(region=-cone)])
        to_cubit_journal(g, world=(500, 500, 500), filename='cone.jou')

@reset_openmc_ids
def test_gq_ellipsoid(request, run_in_tmpdir):
    ellipsoid = openmc.Quadric(1, 2, 3, k=-1)
    g = openmc.Geometry([openmc.Cell(region=-ellipsoid)])
    to_cubit_journal(g, world=(500, 500, 500), filename='ellipsoid.jou')
    diff_gold_file('ellipsoid.jou')
