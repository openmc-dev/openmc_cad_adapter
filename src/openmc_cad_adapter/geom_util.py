import math


def vector_to_euler_xyz(v):
    x, y, z = v
    phi = math.atan2(z, x)
    theta = math.acos(x / math.sqrt(x**2 + y**2))
    psi = math.atan2(y * math.cos(theta), x)

    # Ensure angles are within [0, 2*pi] range
    phi %= (2 * math.pi)
    theta %= (2 * math.pi)
    psi %= (2 * math.pi)

    oe = 180 / math.pi
    return phi * oe, theta * oe, psi * oe


def rotate(id, x, y, z, cmds):
    if nonzero(x, y, z):
        phi, theta, psi = vector_to_euler_xyz((x, y, z))
        cmds.append(f"body {{ {id} }} rotate {phi} about Z")
        cmds.append(f"body {{ {id} }} rotate {theta} about Y")
        cmds.append(f"body {{ {id} }} rotate {psi} about X")


def nonzero(*args):
    return any(arg != 0 for arg in args)


def move( id, x, y, z, cmds):
    if nonzero( x, y, z ):
        cmds.append(f"body {{ {id} }} move {x} {y} {z}")