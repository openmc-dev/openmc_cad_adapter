
import pytest

import openmc

from openmc_cad_adapter import to_cubit_journal
from openmc_cad_adapter.gqs import *

def test_ellipsoid_classification():
    # ELLIPSOID
    testEllip = openmc.Quadric(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    quadric_type, A, B, C, K, _, _ = characterize_general_quadratic(testEllip)
    assert quadric_type == ELLIPSOID


def test_one_sheet_hyperboloid_classification():
    # ONE_SHEET_HYPERBOLOID
    testOneSheet = openmc.Quadric(1.0, 1.0, -1.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    quadric_type, A, B, C, K, _, _ = characterize_general_quadratic(testOneSheet)
    assert quadric_type == ONE_SHEET_HYPERBOLOID


def test_two_sheet_hyperboloid_classification():
    # TWO_SHEET_HYPERBOLOID
    testTwoSheet = openmc.Quadric(-1.0, -1.0, 1.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    quadric_type, A, B, C, K, _, _ = characterize_general_quadratic(testTwoSheet)
    assert quadric_type == TWO_SHEET_HYPERBOLOID


def test_elliptic_cone_classification():
    # ELLIPTIC_CONE
    testEllCone = openmc.Quadric(1.0, 1.0, -1.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    quadric_type, A, B, C, K, _, _ = characterize_general_quadratic(testEllCone)
    assert quadric_type == ELLIPTIC_CONE


def test_elliptic_paraboloid_classification():
    # ELLIPTIC_PARABOLOID
    testEllPara = openmc.Quadric(1.0, 1.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    quadric_type, A, B, C, K, _, _ = characterize_general_quadratic(testEllPara)
    assert quadric_type == ELLIPTIC_PARABOLOID


def test_hyperbolic_paraboloid_classification():
    # HYPERBOLIC_PARABOLOID
    testHypPara = openmc.Quadric(1.0, -1.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    quadric_type, A, B, C, K, _, _ = characterize_general_quadratic(testHypPara)
    assert quadric_type == HYPERBOLIC_PARABOLOID


def test_elliptic_cyl_classification():
    # ELLIPTIC_CYL
    testEllCyl = openmc.Quadric(1.0, 1.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    quadric_type, A, B, C, K, _, _ = characterize_general_quadratic(testEllCyl)
    assert quadric_type == ELLIPTIC_CYLINDER


def test_hyperbolic_cyl_classification():
    # HYPERBOLIC_CYL
    testHypCyl = openmc.Quadric(1.0, -1.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    quadric_type, A, B, C, K, _, _ = characterize_general_quadratic(testHypCyl)
    assert quadric_type == HYPERBOLIC_CYLINDER


def test_parabolic_cyl_classification():
    # PARABOLIC_CYL
    testParaCyl = openmc.Quadric(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0)
    quadric_type, A, B, C, K, _, _ = characterize_general_quadratic(testParaCyl)
    assert quadric_type == PARABOLIC_CYLINDER


# Transformation Tests
def test_ellipsoid_classification():
    # ELLIPSOID
    testRotEllip = openmc.Quadric(103, 125, 66, -48, -12, -60, 0, 0, 0, -294)
    quadric_type, A, B, C, K, _, _ = characterize_general_quadratic(testRotEllip)
    assert quadric_type == ELLIPSOID


def test_elliptic_cone_classification():
    # ELLIPTIC_CONE
    testRotCone = openmc.Quadric(3, 3, -1, 2,  0, 0, 0, 0, 0, 0)
    quadric_type, A, B, C, K, _, _ = characterize_general_quadratic(testRotCone)
    assert quadric_type == ELLIPTIC_CONE


def test_elliptic_paraboloid_classification():
    # ELLIPTIC_PARABOLOID
    testRotEllParab = openmc.Quadric(1, 3, 1, 2, 2, 2, -2, 4, 2, 12)
    quadric_type, A, B, C, K, _, _ = characterize_general_quadratic(testRotEllParab)
    assert quadric_type == ELLIPTIC_PARABOLOID


def test_elliptic_cylinder_classification():
    # ELLIPTIC_CYLINDER
    testRotEllCyl = openmc.Quadric(5, 2, 5, -4, -4, -2, 6, -12, 18, -3)
    quadric_type, A, B, C, K, _, _ = characterize_general_quadratic(testRotEllCyl)
    assert quadric_type == ELLIPTIC_CYLINDER


def test_parabolic_cylinder_classification():
    # PARABOLIC CYLINDER
    testRotParaCyl = openmc.Quadric(9, 36, 4, -36, -24, 12, -16, -24, -48, 56)
    quadric_type, A, B, C, K, _, _ = characterize_general_quadratic(testRotParaCyl)
    assert quadric_type == PARABOLIC_CYLINDER