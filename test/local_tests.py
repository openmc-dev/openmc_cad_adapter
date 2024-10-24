import functools

import openmc

from openmc_cad_adapter import to_cubit_journal

from .test_utilities import diff_gold_file


def openmc_reset(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        openmc.reset_auto_ids()
        func(*args, **kwargs)
    return wrapper


@openmc_reset
def test_xcylinder(request):
    x_cyl = openmc.XCylinder(r=1.0, y0=10.0, z0=5.0)
    g = openmc.Geometry([openmc.Cell(region=-x_cyl)])
    to_cubit_journal(g, world=(500, 500, 500), filename='xcylinder.jou')
    diff_gold_file('xcylinder.jou')


@openmc_reset
def test_ycylinder(request):
    y_cyl = openmc.YCylinder(r=1.0, x0=10.0, z0=5.0)
    g = openmc.Geometry([openmc.Cell(region=-y_cyl)])
    to_cubit_journal(g, world=(500, 500, 500), filename='ycylinder.jou')
    diff_gold_file('ycylinder.jou')


