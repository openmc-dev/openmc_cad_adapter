import math
import numpy as np

def vector_to_euler_xyz(v):
    v = np.asarray(v)
    v /= np.linalg.norm(v)

    x, y, z = v

    xy_norm = math.sqrt(x**2 + y**2)

    theta = np.arctan2(z, xy_norm)
    # if z component is zero, vector is in the xy plane
    if z == 0:
        theta = np.pi / 2
    phi = np.arctan2(y, x)

    # Ensure angles are within [0, 2*pi] range
    phi %= (2 * math.pi)
    theta %= (2 * math.pi)

    oe = 180 / math.pi
    return phi * oe, theta * oe, 0.0


def rotate(id, x, y, z, cmds):
    if nonzero(x, y, z):
        phi, theta, psi = vector_to_euler_xyz((x, y, z))
        cmds.append(f"body {{ {id} }} rotate {theta} about Y")
        cmds.append(f"body {{ {id} }} rotate {phi} about Z")
        # cmds.append(f"body {{ {id} }} rotate {phi} about Z")
        # cmds.append(f"body {{ {id} }} rotate {theta} about Y")
        # cmds.append(f"body {{ {id} }} rotate {psi} about X")


def nonzero(*args):
    return any(arg != 0 for arg in args)


def move( id, x, y, z, cmds):
    if nonzero( x, y, z ):
        cmds.append(f"body {{ {id} }} move {x} {y} {z}")