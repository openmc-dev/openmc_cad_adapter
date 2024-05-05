import math
import sys
from openmc.region import Region, Complement, Intersection, Union
from openmc.surface import Halfspace, Quadric
from openmc.lattice import Lattice, HexLattice
from numpy.linalg import matrix_rank
import numpy as np
            
ELLIPSOID = 1
ONE_SHEET_HYPERBOLOID = 2
TWO_SHEET_HYPERBOLOID = 3
ELLIPTIC_CONE = 4
ELLIPTIC_PARABOLOID = 5
HYPERBOLIC_PARABOLOID = 6
ELLIPTIC_CYLINDER = 7
HYPERBOLIC_CYLINDER = 8
PARABOLIC_CYLINDER = 9

def characterize_general_quadratic( surface ): #s surface
    gq_tol = 1e-6
    equivalence_tol = 1e-8
    a = surface.coefficients['a']  
    b = surface.coefficients['b']  
    c = surface.coefficients['c']  
    d = surface.coefficients['d']  
    e = surface.coefficients['e']  
    f = surface.coefficients['f']  
    g = surface.coefficients['g']  
    h = surface.coefficients['h']  
    j = surface.coefficients['j']  
    k = surface.coefficients['k']  
    #coefficient matrix
    Aa = np.matrix([
               [a, d/2, f/2], 
               [d/2, b, e/2],
               [f/2, e/2, c]])
    #hessian matrix
    Ac = np.matrix([
               [a, d/2, f/2, g/2], 
               [d/2, b, e/2, h/2],
               [f/2, e/2, c, j/2],
               [g/2, h/2, j/2, k]])
    rank_Aa = matrix_rank( Aa )
    rank_Ac = matrix_rank( Ac )
    
    det_Ac = np.linalg.det(Ac)
    if np.abs( det_Ac ) < gq_tol:
        delta = 0
    else:
        delta = -1 if det_Ac < 0 else -1
    eigen_results = np.linalg.eig(Aa);
    signs = np.array([ 0, 0, 0 ])
    for i in range( 0, 3 ):
        if eigen_results.eigenvalues[ i ] > -1 * gq_tol:
            signs[i] = 1
        else:
            signs[i] = -1

    S = 1 if np.abs( signs.sum() ) == 3 else -1

    B = np.array([[ -g/2], [-h/2], [-j/2 ]])

    Aai = np.linalg.pinv( Aa )

    C = Aai * B

    dx = C[0]
    dy = C[1]
    dz = C[2]

    #Update the constant using the resulting translation
    K_ = k + g/2*dx + h/2*dy + j/2*dz

    if rank_Aa == 2 and rank_Ac == 3 and S == 1:
        delta = -1 if K_ * signs[0] else 1

    D = -1 if K_ * signs[0] else 1


    def find_type( rAa, rAc, delta, S, D ):
        if 3 == rAa and 4 == rAc and -1 == delta and 1 == S:
            return ELLIPSOID
        elif 3 == rAa and 4 == rAc and 1 == delta and -1 == S:
            return ONE_SHEET_HYPERBOLOID
        elif 3 == rAa and 4 == rAc and -1 == delta and -1 == S:
            return TWO_SHEET_HYPERBOLOID
        elif 3 == rAa and 3 == rAc and 0 == delta and -1 == S:
            return ELLIPTIC_CONE
        elif 2 == rAa and 4 == rAc and -1 == delta and 1 == S:
            return ELLIPTIC_PARABOLOID
        elif 2 == rAa and 4 == rAc and 1 == delta and -1 == S:
            return HYPERBOLIC_PARABOLOID
        elif 2 == rAa and 3 == rAc and -1 == delta and 1 == S:
            return ELLIPTIC_CYLINDER
        elif 2 == rAa and 3 == rAc and 0 == delta and -1 == S:
            return HYPERBOLIC_CYLINDER
        elif 2 == rAa and 3 == rAc and 0 == delta and 1 == S:
            return PARABOLIC_CYLINDER
        elif 2 == rAa and 3 == rAc and 1 == S and D != 0 :
            return find_type( rAa, rAc, D, S, 0 )
        else:
            raise "UNKNOWN QUADRATIC"
        
    gq_type = find_type( rank_Aa, rank_Ac, delta, S, D )
    
    #set the translation
    translation = C

    rotation_matrix = eigen_results.eigenvectors
    eigenvalues = eigen_results.eigenvalues
    
    for i in range( 0, 3 ):
        if abs(eigenvalues[i]) < gq_tol:
            eigenvalues[i] = 0
        
    A_ = eigenvalues[0]
    B_ = eigenvalues[1]
    C_ = eigenvalues[2];
    D_ = 0; E_ = 0; F_ = 0;
    G_ = 0; H_ = 0; J_ = 0;
    if gq_type == ONE_SHEET_HYPERBOLOID:
        if abs( K_) < equivalence_tol:
           K_ = 0
           return ELLIPTIC_CONE
    if gq_type == TWO_SHEET_HYPERBOLOID:
        if abs( K_) < equivalence_tol:
           K_ = 0
           return ELLIPTIC_CONE
    if gq_type == ELLIPSOID:
        if abs( A_) < equivalence_tol:
           A_ = 0
           return ELLIPTIC_CYLINDER
        elif abs( B_) < equivalence_tol:
           B_ = 0
           return ELLIPTIC_CYLINDER
        elif abs( C_) < equivalence_tol:
           C_ = 0
           return ELLIPTIC_CYLINDER
    else:
        return ( gq_type, A_, B_, C_, K_, translation, rotation_matrix )


    
def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])

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

def to_cubit_journal(geom, seen=set(), world=[60,60,60], cells=None, filename=None, to_cubit=False ):
    w = world
    cid = 1
    def lastid():
        nonlocal cid 
        id = cid
        cid = cid + 1
        return id
    cmds = []
    cmds.extend( [
        #"set echo off",
        "set graphics off",
        "set journal off",
        #"set undo off",
        #"set default autosizing off",   # especially when the CAD is complex (contains many spline surfaces) this can have a massive effect
        #"set info off",
        #"set warning off",
        ])
    def python_cmd( s ):
        cmds.append( s )
    def cubit_cmd( s ):
        cmds.append( s )
        #cmds.append( f'cubit.cmd( "{s}" )' )

    def new_variable( ):
        idn = lastid()
        return f"id{idn}"

    def emit_get_last_id( type = "body" ):
        idn = lastid()
        ids = f"id{idn}"
        python_cmd( f'#{{ {ids} = Id("{type}") }}' )
        return ids

    def rotate( id, x, y, z ):
        if nonzero( x, y, z ):
            phi, theta, psi = vector_to_euler_xyz( ( x, y, z ) )
            cubit_cmd( f"body {{ {id} }} rotate {phi} about Z" )
            cubit_cmd( f"body {{ {id} }} rotate {theta} about Y" )
            cubit_cmd( f"body {{ {id} }} rotate {psi} about X" )

    def nonzero(*args):
        return any(arg!= 0 for arg in args)

    def move( id, x, y, z ):
        if nonzero( x, y, z ):
           cubit_cmd( f"body {{ {id} }} move {x} {y} {z}" )

    def make_world_brick():
        pass
        
    def surface_to_cubit_journal(node, w, indent = 0, inner_world = None, hex = False, ent_type = "body" ):
        def ind():
            return ' ' * (2*indent)
        if isinstance(node, Halfspace):
            #if not ( node.surface in seen ):
                seen.add( node.surface )
                surface = node.surface
               #print( ind(), "Surface:", surface )

                def reverse():
                    if node.side == '-':
                        #print( "REV", node, node.side )
                        return "reverse"
                    else:
                        #print( "NORMAL", node, node.side )
                        return ""

                if surface._type == "plane":
                    ca = surface.coefficients['a']
                    cb = surface.coefficients['b']
                    cc = surface.coefficients['c']
                    cd = surface.coefficients['d']
                    n = np.array([ca, cb, cc ])
                    maxi = 0
                    maxv = sys.float_info.min 
                    for i in range( 0, 3 ):
                        if n[i] > maxv:
                            maxi = i
                            maxv = n[i]

                    ns = cd * n

                    if False:
                        n2 = np.copy( n )
                        n2[ maxi ] = - n2[maxi]
                        n3 = np.cross(n, n2)
                        n4 = np.cross(n, n3)
                        print( n2, n3, n4 )
                        n3 = n3/np.linalg.norm(n3)
                        n4 = n4/np.linalg.norm(n4)
                        n3 = np.add(ns, n3)
                        n4 = np.add(ns, n4)
                        cmds.append( f"create vertex {ns[0]} {ns[1]} {ns[2]}")
                        v1 = emit_get_last_id( "vertex" )
                        cmds.append( f"create vertex {n3[0]} {n3[1]} {n3[2]}")
                        v2 = emit_get_last_id( "vertex" )
                        cmds.append( f"create vertex {n4[0]} {n4[1]} {n4[2]}")
                        v3 = emit_get_last_id( "vertex" )
                        cmds.append( f"create planar surface with plane vertex {{ {v1} }} vertex {{ {v2} }} vertex {{ {v3} }} Extended Absolute {w[0]}")
                        surf = emit_get_last_id( "surface" )
                        cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                        ids = emit_get_last_id( ent_type )
                        cmds.append( f"section body {{ {ids} }} with surface {{ {surf} }} {reverse()}")
                        cmds.append( f"del surface {{ {surf} }}")
                        cmds.append( f"del vertex {{ {v1} }} {{ {v2} }} {{ {v3} }}")
                    else:
                        #for some reason cubit sectioning with a surface is backwards from cubit sectioning with an axis aligned plane
                        def lreverse():
                            #print( node.side)
                            if node.side == '-':
                                    return ""
                            else:
                                    return "reverse"

                        #print( ca, cb, cc, cd )
                        cmds.append( f"create surface rectangle width  { 2*w[0] } zplane")
                        sur = emit_get_last_id( "surface" )
                        surf = emit_get_last_id( "body" )
                        cmds.append( f"# dir {node.side}" )
                        cmds.append( f"# c_  {ca} {cb} {cc} {cd}" )
                        #cmds.append( f"# n  {n[0]} {n[1]} {n[2]}" )
                        n = n/np.linalg.norm(n)
                        ns = cd * n
                        #cmds.append( f"# n  {n[0]} {n[1]} {n[2]}" )
                        zn = np.array([ 0, 0, 1 ])
                        n3 = np.cross(n, zn )
                        dot = np.dot(n, zn )
                        cmds.append( f"# n3 {n3[0]} {n3[1]} {n3[2]}" )
                        #cmds.append( f"# dot {dot}" )
                        degs = math.degrees( math.acos( np.dot( n, zn ) ) )
                        y = np.linalg.norm( n3 )
                        x = dot
                        #cubit 'rotate body' introduces a negative so we compensate here with a negative
                        angle = - math.degrees( math.atan2( y, x ) )
                        #print( n3 )
                        #print(  math.acos( np.dot( n, zn ) ),  math.degrees( math.acos( np.dot( n, zn ) ) ) )
                        if n3[0] != 0 or n3[1] != 0 or n3[2] != 0:
                            cmds.append( f"Rotate body {{ {surf} }} about 0 0 0 direction {n3[0]} {n3[1]} {n3[2]} Angle {angle}")
                        cmds.append( f"body {{ { surf } }} move {ns[0]} {ns[1]} {ns[2]}")
                        cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                        ids = emit_get_last_id( ent_type )
                        cmds.append( f"section body {{ {ids} }} with surface {{ {sur} }} {lreverse()}")
                        cmds.append( f"del surface {{ {sur} }}")


                    # phi, theta, psi = vector_to_euler_xyz( ( surface.coefficients['a'], surface.coefficients['b'], surface.coefficients['c'] ) )
                    # cmds.append( f"body {{ { ids } }} rotate {phi} about Z" )
                    # cmds.append( f"body {{ { ids } }} rotate {theta} about Y" )
                    # cmds.append( f"body {{ { ids } }} rotate {psi} about X" )
                    # ca = surface.coefficients['a']
                    # cb = surface.coefficients['b']
                    # cc = surface.coefficients['c']
                    # cd = surface.coefficients['d']
                    # n = np.array([ca, cb, cc ])
                    # n_length = np.linalg.norm(n)
                    # dd = cd / n_length 
                    # print( phi, theta, psi )
                    # print( ca, cb, cc, cd, n_length, dd )
                    # if node.side == '-':
                    #     dd = dd - w[0]/2
                    # cmds.append( f"body {{ { ids } }} move direction {ca} {cb} {cc} distance {dd}" )
                    return ids
                elif surface._type == "x-plane":
                    cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                    ids = emit_get_last_id( ent_type )
                    cmds.append( f"section body {{ {ids} }} with xplane offset {surface.coefficients['x0']} {reverse()}")
                    #ids = emit_get_last_id()
                    return ids
                elif surface._type == "y-plane":
                    cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                    ids = emit_get_last_id( ent_type )
                    cmds.append( f"section body {{ {ids} }} with yplane offset {surface.coefficients['y0']} {reverse()}")
                    #ids = emit_get_last_id()
                    return ids
                elif surface._type == "z-plane":
                    cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                    ids = emit_get_last_id( ent_type )
                    cmds.append( f"section body {{ {ids} }} with zplane offset {surface.coefficients['z0']} {reverse()}")
                    #ids = emit_get_last_id()
                    return ids
                elif surface._type == "cylinder":
                    h = inner_world[2] if inner_world else w[2] 
                    cmds.append( f"cylinder height {h} radius {surface.coefficients['r']}")
                    ids = emit_get_last_id()
                    if node.side != '-':
                        wid = 0
                        if inner_world:
                            if hex:
                                cmds.append( f"create prism height {inner_world[2]} sides 6 radius { ( inner_world[0] / 2 ) }" )
                                wid = emit_get_last_id( ent_type )
                                cmds.append( f"rotate body {{ {wid} }} about z angle 30" )
                            else:
                                cmds.append( f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}" )
                                wid = emit_get_last_id( ent_type )
                        else:
                            cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                            wid = emit_get_last_id( ent_type )
                        cmds.append( f"subtract body {{ { ids } }} from body {{ { wid } }}" )
                        rotate( wid, surface.coefficients['dx'], surface.coefficients['dy'], surface.coefficients['dz'] )
                        move( wid, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'] )
                        return wid
                    rotate( ids, surface.coefficients['dx'], surface.coefficients['dy'], surface.coefficients['dz'] )
                    move( ids, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'] )
                    return ids
                elif surface._type == "x-cylinder":
                    h = inner_world[0] if inner_world else w[0] 
                    cmds.append( f"cylinder height {h} radius {surface.coefficients['r']}")
                    ids = emit_get_last_id( ent_type )
                    cmds.append( f"rotate body {{ { ids } }} about y angle 90")
                    if node.side != '-':
                        wid = 0
                        if inner_world:
                            if hex:
                                cmds.append( f"create prism height {inner_world[2]} sides 6 radius { inner_world[0] / 2 } " )
                                wid = emit_get_last_id( ent_type )
                                cmds.append( f"rotate body {{ {wid} }} about z angle 30" )
                                cmds.append( f"rotate body {{ {wid} }} about y angle 90")
                            else:
                                cmds.append( f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}" )
                                wid = emit_get_last_id( ent_type )
                        else:
                            cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                            wid = emit_get_last_id( ent_type )
                        cmds.append( f"subtract body {{ { ids } }} from body {{ { wid } }}" )
                        move( wid, 0, surface.coefficients['y0'], surface.coefficients['z0'] )
                        return wid
                    move( ids, 0, surface.coefficients['y0'], surface.coefficients['z0'] )
                    return ids
                elif surface._type == "y-cylinder":
                    h = inner_world[1] if inner_world else w[1] 
                    cmds.append( f"cylinder height {h} radius {surface.coefficients['r']}")
                    ids = emit_get_last_id( ent_type )
                    cmds.append( f"rotate body {{ {ids} }} about x angle 90")
                    if node.side != '-':
                        wid = 0
                        if inner_world:
                            if hex:
                                cmds.append( f"create prism height {inner_world[2]} sides 6 radius { ( inner_world[0] / 2) }" )
                                wid = emit_get_last_id( ent_type )
                                cmds.append( f"rotate body {{ {wid} }} about z angle 30" )
                                cmds.append( f"rotate body {{ {wid} }} about x angle 90")
                            else:
                                cmds.append( f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}" )
                                wid = emit_get_last_id( ent_type )
                        else:
                            cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                            wid = emit_get_last_id( ent_type )
                        cmds.append( f"subtract body {{ {ids} }} from body {{ {wid} }}" )
                        move( wid, surface.coefficients['x0'], 0, surface.coefficients['z0'] )
                        return wid
                    move( ids, surface.coefficients['x0'], 0, surface.coefficients['z0'] )
                    return ids
                elif surface._type == "z-cylinder":
                    h = inner_world[2] if inner_world else w[2] 
                    cmds.append( f"cylinder height {h} radius {surface.coefficients['r']}")
                    ids = emit_get_last_id( ent_type )
                    if node.side != '-':
                        wid = 0
                        if inner_world:
                            if hex:
                                cmds.append( f"create prism height {inner_world[2]} sides 6 radius { ( inner_world[0] / 2 ) }" )
                                wid = emit_get_last_id( ent_type )
                                cmds.append( f"rotate body {{ {wid} }} about z angle 30" )
                            else:
                                cmds.append( f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}" )
                                wid = emit_get_last_id( ent_type )
                        else:
                            cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                            wid = emit_get_last_id( ent_type )
                        cmds.append( f"subtract body {{ { ids } }} from body {{ { wid } }}" )
                        move( wid, surface.coefficients['x0'], surface.coefficients['y0'], 0 )
                        return wid
                    move( ids, surface.coefficients['x0'], surface.coefficients['y0'], 0 )
                    return ids
                elif surface._type == "sphere":
                    cmds.append( f"sphere redius {surface.coefficients['r']}")
                    ids = emit_get_last_id( ent_type )
                    move( ids, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['zy0'] )
                    pass
                elif surface._type == "cone":
                    raise "cone not implemented"
                    pass
                elif surface._type == "x-cone":
                    cmds.append( f"create frustum height {w[0]} radius {math.sqrt(surface.coefficients['r2']*w[0])} top 0")
                    ids = emit_get_last_id( ent_type )
                    cmds.append( f"rotate body {{ {ids} }} about y angle 90")
                    if node.side != '-':
                        cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                        wid = emit_get_last_id( ent_type )
                        cmds.append( f"subtract body {{ {ids} }} from body {{ {wid} }}" )
                        move( wid, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'] )
                        return wid
                    move( ids, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'] )
                    return ids
                elif surface._type == "y-cone":
                    cmds.append( f"create frustum height {w[1]} radius {math.sqrt(surface.coefficients['r2']*w[1])} top 0")
                    ids = emit_get_last_id( ent_type )
                    cmds.append( f"rotate body {{ {ids} }} about x angle 90")
                    if node.side != '-':
                        cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                        wid = emit_get_last_id( ent_type )
                        cmds.append( f"subtract body {{ {ids} }} from body {{ {wid} }}" )
                        move( wid, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'] )
                        return wid
                    move( ids, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'] )
                    return ids
                elif surface._type == "z-cone":
                    cmds.append( f"create frustum height {w[2]} radius {math.sqrt(surface.coefficients['r2']*w[2])} top 0")
                    ids = emit_get_last_id( ent_type )
                    if node.side != '-':
                        cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                        wid = emit_get_last_id( ent_type )
                        cmds.append( f"subtract body {{ {ids} }} from body {{ {wid} }}" )
                        move( wid, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'] )
                        return wid
                    move( ids, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'] )
                    return ids
                elif surface._type == "x-torus":
                    cmds.append( f"torus major radius {surface.coefficients['r']} minor radius {surface.coefficients['r']}")
                    ids = emit_get_last_id( ent_type )
                    cmds.append( f"rotate body {{ {ids} }} about y angle 90")
                    if node.side != '-':
                        cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                        wid = emit_get_last_id( ent_type )
                        cmds.append( f"subtract body {{ {ids} }} from body {{ {wid} }}" )
                        move( wid, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'] )
                        return wid
                    move( ids, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'] )
                    return ids
                elif surface._type == "y-torus":
                    cmds.append( f"torus major radius {surface.coefficients['r']} minor radius {surface.coefficients['r']}")
                    ids = emit_get_last_id( ent_type )
                    cmds.append( f"rotate body {{ {ids} }} about x angle 90")
                    if node.side != '-':
                        cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                        wid = emit_get_last_id( ent_type )
                        cmds.append( f"subtract body {{ {id} }} from body {{ {wid} }}" )
                        move( wid, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'] )
                        return wid
                    return ids
                elif surface._type == "z-torus":
                    cmds.append( f"torus major radius {surface.coefficients['r']} minor radius {surface.coefficients['r']}")
                    ids = emit_get_last_id( ent_type )
                    if node.side != '-':
                        cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                        wid = emit_get_last_id( ent_type )
                        cmds.append( f"subtract body {{ {ids} }} from body {{ {wid} }}" )
                        move( wid, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'] )
                        return wid
                    move( ids, surface.coefficients['x0'], surface.coefficients['y0'], surface.coefficients['z0'] )
                    return ids
                elif surface._type == "quadric":
                    ( gq_type, A_, B_, C_, K_, translation, rotation_matrix ) = characterize_general_quadratic( surface )

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

                          axis[col] = math.sqrt( (mat[col,col]+1)/2 );
                          denom = 2*axis[col];
                          axis[(col+1)%3] = mat[col,(col+1)%3] / denom;
                          axis[(col+2)%3] = mat[col,(col+2)%3] / denom;
                          return ( axis, theta )
                        else:
                          axis = np.array([ x/r, y/r, z/r ])
                          return ( axis, theta )

                    #print( "Quadric", characterize_general_quadratic( surface ) )
                    #print( gq_type, A_, B_, C_, K_ )
                    #print( translation )
                    #print( rotation_matrix )
                    (r_axis, r_theta ) = rotation_to_axis_angle( rotation_matrix )
                    #compensate for cubits insertion of a negative
                    r_degs = - math.degrees( r_theta )
                    print( r_axis, math.degrees( r_theta ), r_degs )
                    if gq_type == ELLIPSOID : #1
                            r1 = math.sqrt( abs( -K_/A_ ) )
                            r2 = math.sqrt( abs( -K_/B_ ) )
                            r3 = math.sqrt( abs( -K_/C_ ) )
                            cmds.append( f"sphere redius 1")
                            ids = emit_get_last_id( ent_type )
                            cmds.append( f"body {{ { ids } }} scale x { r1 } y { r2 } z { r3 }")
                            move( ids, translation[0,0], translation[1,0], translation[2,0] )
                    elif gq_type == ELLIPTIC_CYLINDER : #7
                        if A_ == 0:
                            print( "X", gq_type, A_, B_, C_, K_, r_axis, r_degs )
                            h = inner_world[0] if inner_world else w[0] 
                            r1 = math.sqrt( abs( K_/C_ ) )
                            r2 = math.sqrt( abs( K_/B_ ) )
                            cmds.append( f"cylinder height {h} Major Radius {r1} Minor Radius {r2}")
                            ids = emit_get_last_id( ent_type )
                            cmds.append( f"rotate body {{ { ids } }} about y angle 90")
                            if node.side != '-':
                                wid = 0
                                if inner_world:
                                    if hex:
                                        cmds.append( f"create prism height {inner_world[2]} sides 6 radius { inner_world[0] / 2 } " )
                                        wid = emit_get_last_id( ent_type )
                                        cmds.append( f"rotate body {{ {wid} }} about z angle 30" )
                                        cmds.append( f"rotate body {{ {wid} }} about y angle 90")
                                    else:
                                        cmds.append( f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}" )
                                        wid = emit_get_last_id( ent_type )
                                else:
                                    cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                                    wid = emit_get_last_id( ent_type )
                                cmds.append( f"subtract body {{ { ids } }} from body {{ { wid } }}" )
                                cmds.append( f"Rotate body {{ {wid } }} about 0 0 0 direction {r_axis[0]} {r_axis[1]} {r_axis[2]} Angle {r_degs}")
                                move( wid, translation[0,0], translation[1,0], translation[2,0] )
                                return wid
                            cmds.append( f"Rotate body {{ {ids} }} about 0 0 0 direction {r_axis[0]} {r_axis[1]} {r_axis[2]} Angle {r_degs}")
                            move( ids, translation[0,0], translation[1,0], translation[2,0] )
                            return ids
                        if B_ == 0:
                            print( "Y", gq_type, A_, B_, C_, K_ )
                            h = inner_world[1] if inner_world else w[1] 
                            r1 = math.sqrt( abs( K_/A_ ) )
                            r2 = math.sqrt( abs( K_/C_ ) )
                            cmds.append( f"cylinder height {h} Major Radius {r1} Minor Radius {r2}")
                            ids = emit_get_last_id( ent_type )
                            cmds.append( f"rotate body {{ { ids } }} about x angle 90")
                            if node.side != '-':
                                wid = 0
                                if inner_world:
                                    if hex:
                                        cmds.append( f"create prism height {inner_world[2]} sides 6 radius { inner_world[0] / 2 } " )
                                        wid = emit_get_last_id( ent_type )
                                        cmds.append( f"rotate body {{ {wid} }} about z angle 30" )
                                        cmds.append( f"rotate body {{ {wid} }} about y angle 90")
                                    else:
                                        cmds.append( f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}" )
                                        wid = emit_get_last_id( ent_type )
                                else:
                                    cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                                    wid = emit_get_last_id( ent_type )
                                cmds.append( f"subtract body {{ { ids } }} from body {{ { wid } }}" )
                                cmds.append( f"Rotate body {{ {wid } }} about 0 0 0 direction {r_axis[0]} {r_axis[1]} {r_axis[2]} Angle {r_degs}")
                                move( wid, translation[0,0], translation[1,0], translation[2,0] )
                                return wid
                            cmds.append( f"Rotate body {{ {ids} }} about 0 0 0 direction {r_axis[0]} {r_axis[1]} {r_axis[2]} Angle {r_degs}")
                            move( ids, translation[0,0], translation[1,0], translation[2,0] )
                            return ids
                        if C_ == 0:
                            print( "Z", gq_type, A_, B_, C_, K_ )
                            h = inner_world[2] if inner_world else w[2] 
                            r1 = math.sqrt( abs( K_/A_ ) )
                            r2 = math.sqrt( abs( K_/B_ ) )
                            cmds.append( f"cylinder height {h} Major Radius {r1} Minor Radius {r2}")
                            ids = emit_get_last_id( ent_type )
                            if node.side != '-':
                                wid = 0
                                if inner_world:
                                    if hex:
                                        cmds.append( f"create prism height {inner_world[2]} sides 6 radius { inner_world[0] / 2 } " )
                                        wid = emit_get_last_id( ent_type )
                                        cmds.append( f"rotate body {{ {wid} }} about z angle 30" )
                                        cmds.append( f"rotate body {{ {wid} }} about y angle 90")
                                    else:
                                        cmds.append( f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}" )
                                        wid = emit_get_last_id( ent_type )
                                else:
                                    cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
                                    wid = emit_get_last_id( ent_type )
                                cmds.append( f"subtract body {{ { ids } }} from body {{ { wid } }}" )
                                cmds.append( f"Rotate body {{ {wid } }} about 0 0 0 direction {r_axis[0]} {r_axis[1]} {r_axis[2]} Angle {r_degs}")
                                move( wid, translation[0,0], translation[1,0], translation[2,0] )
                                return wid
                            cmds.append( f"Rotate body {{ {ids} }} about 0 0 0 direction {r_axis[0]} {r_axis[1]} {r_axis[2]} Angle {r_degs}")
                            move( ids, translation[0,0], translation[1,0], translation[2,0] )
                            return ids
                    elif gq_type == ELLIPTIC_CONE : #3
                        if A_ == 0:
                            #print( "X", gq_type, A_, B_, C_, K_ )
                            h = inner_world[0] if inner_world else w[0] 
                            minor = math.sqrt( abs( -A_/C_ ) )
                            major = math.sqrt( abs( -A_/B_ ) )
                            rot_angle = - 90
                            rot_axis = 1
                            cmds.append( f"create frustum height {h} Major Radius {major} Minor Radius {minor} top 0")
                            ids = emit_get_last_id( ent_type )
                            cmds.append( f"rotate body {{ { ids } }} about y angle -90")
                            cmds.append( f"copy body {{ { ids } }}") 
                            mirror = emit_get_last_id( ent_type )
                            cmds.append( f"rotate body {{ { mirror } }} about 0 0 0 angle 180")
                            cmds.append( f"unit body {{ { ids } }} {{ { mirror } }}")
                            cmds.append( f"Rotate body {{ {ids} }} about 0 0 0 direction {r_axis[0]} {r_axis[1]} {r_axis[2]} Angle {r_degs}")
                            move( ids, translation[0,0], translation[1,0], translation[2,0] )
                            return ids
                        if B_ == 0:
                            #print( "Y", gq_type, A_, B_, C_, K_ )
                            h = inner_world[1] if inner_world else w[1] 
                            minor = math.sqrt( abs( -B_/A_ ) )
                            major = math.sqrt( abs( -B_/C_ ) )
                            rot_angle = 90
                            rot_axis = 0
                            cmds.append( f"create frustum height {h} Major Radius {major} Minor Radius {minor} top 0")
                            ids = emit_get_last_id( ent_type )
                            cmds.append( f"rotate body {{ { ids } }} about x angle 90")
                            cmds.append( f"copy body {{ { ids } }}") 
                            mirror = emit_get_last_id( ent_type )
                            cmds.append( f"rotate body {{ { mirror } }} about 0 0 0 angle 180")
                            cmds.append( f"unit body {{ { ids } }} {{ { mirror } }}")
                            cmds.append( f"Rotate body {{ {ids} }} about 0 0 0 direction {r_axis[0]} {r_axis[1]} {r_axis[2]} Angle {r_degs}")
                            move( ids, translation[0,0], translation[1,0], translation[2,0] )
                            return ids
                        if C_ == 0:
                            #print( "Z", gq_type, A_, B_, C_, K_ )
                            h = inner_world[2] if inner_world else w[2] 
                            minor = math.sqrt( abs( -C_/A_ ) )
                            major = math.sqrt( abs( -C_/B_ ) )
                            rot_angle = 180 
                            rot_axis = 0
                            cmds.append( f"create frustum height {h} Major Radius {major} Minor Radius {minor} top 0")
                            ids = emit_get_last_id( ent_type )
                            cmds.append( f"copy body {{ { ids } }}") 
                            mirror = emit_get_last_id( ent_type )
                            cmds.append( f"rotate body {{ { mirror } }} about 0 0 0 angle 180")
                            cmds.append( f"unit body {{ { ids } }} {{ { mirror } }}")
                            cmds.append( f"Rotate body {{ {ids} }} about 0 0 0 direction {r_axis[0]} {r_axis[1]} {r_axis[2]} Angle {r_degs}")
                            move( ids, translation[0,0], translation[1,0], translation[2,0] )
                            return ids
                    else:
                        raise f"{surface.type} not implemented"

                else:
                    print( f"{surface.type} not implemented" )
                    raise f"{surface.type} not implemented"
            #else:
            #    print( ind(), node )

        elif isinstance(node, Complement):
            print( "Complement:" )
            id = surface_to_cubit_journal(node.node, w, indent + 1, inner_world, ent_type = ent_type )
            cmds.append( f"brick x {w[0]} y {w[1]} z {w[2]}" )
            wid = emit_get_last_id( ent_type )
            cmds.append( f"subtract body {{ {id} }} from body {{ {wid} }}" )
            return emit_get_last_id( ent_type )
        elif isinstance(node, Intersection):
            #print( ind(), "Intersection:" )
            last = 0
            if len( node ) > 0:
                last = surface_to_cubit_journal( node[0], w, indent + 1, inner_world, ent_type = ent_type ,)
                for subnode in node[1:]:
                    s = surface_to_cubit_journal( subnode, w, indent + 1, inner_world, ent_type = ent_type ,)
                    before = emit_get_last_id()
                    cmds.append( f"intersect {ent_type} {{ {last} }} {{ {s} }}" )
                    after = emit_get_last_id()
                    last = new_variable();
                    cmds.append( f"#{{{last} = ( {before} == {after} ) ? {s} : {after}}}" )
                    #last = emit_get_last_id( ent_type )
                    #last = s
                if inner_world:
                    cmds.append( f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}" )
                    iwid = emit_get_last_id( ent_type )
                    cmds.append( f"intersect {ent_type} {{ {last} }} {{ {iwid} }}" )
                    return emit_get_last_id( ent_type )
            return last
            #return last
        elif isinstance(node, Union):
            #print( ind(), "Union:" )
            if len( node ) > 0:
                local_ent_type = "body"
                first = surface_to_cubit_journal( node[0], w, indent + 1, inner_world, ent_type = local_ent_type )
                for subnode in node[1:]:
                    s = surface_to_cubit_journal( subnode, w, indent + 1, inner_world, ent_type = local_ent_type )
                    cmds.append( f"unite {local_ent_type} {{ {first} }} {{ {s} }}" )
                if inner_world:
                    cmds.append( f"brick x {inner_world[0]} y {inner_world[1]} z {inner_world[2]}" )
                    iwid = emit_get_last_id( local_ent_type )
                    cmds.append( f"intersect {ent_type} {{ {last} }} {{ {iwid} }}" )
                    return first
            return first
        elif isinstance(node, Quadric):
            pass
        else:
            print( f"{node} not implemented" )
            raise f"{node} not implemented"
            #print( node )
            ##print( ind(), "Composite:", node )
            for subnode in node:
                ##print( ind(), "Subnode:", subnode )
                surface_to_cubit_journal( subnode, w, indent + 1, inner_world, ent_type = ent_type )

    def process_node_or_fill( node, w, indent = 0, offset = [0, 0], inner_world = None, outer_ll = None, ent_type = "body", hex = False ):
        def ind():
            return ' ' * (2*indent)
        #if not ( node in seen ):
        #print(  ind(), "Node:", node )
        seen.add( node )
        results = []
        if hasattr( node, "region" ) and not ( hasattr( node, "fill" ) and isinstance(node.fill, Lattice) ):
            if node.region != None:
                #print(  ind(), "Region:", node.region )
                id = surface_to_cubit_journal( node.region, w, indent, inner_world, hex = hex )
                results.append( id )
            elif hex:
                cmds.append( f"create prism height {inner_world[2]} sides 6 radius { ( inner_world[0] / 2) }" )
                wid = emit_get_last_id( ent_type )
                cmds.append( f"rotate body {{ {wid} }} about z angle 30" )
                results.append( wid )

        if hasattr( node, "fill" ) and isinstance(node.fill, Lattice):
            #print(  ind(), "Fill:", node.fill )
            id = process_node_or_fill( node.fill, w, indent + 1, offset, inner_world )
            results.append( id )
        if hasattr( node, "universes" ):
            #print(  ind(), "Universes:", node )
            pitch = node._pitch

            if isinstance( node, HexLattice ) :
                three_d_hex_lattice = len( node._pitch ) > 1
                #three_d_hex_lattice = len( node.center ) > 2
                #3d hex lattice
                if three_d_hex_lattice:
                    center = [ node.center[0], node.center[1], node.center[1] ]
                    # ll = [ node.lower_left[0], node.lower_left[1] ]
                    # if outer_ll:
                    #     ll = outer_ll
                    # ll[0] = ll[0] + pitch[0] / 2
                    # ll[1] = ll[1] + pitch[1] / 2
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
                            #print( "diameter", pitch[0] )
                            #print( "side_to_side_diameter", side_to_side_diameter )
                            #print( "center_to_mid_side_diameter", center_to_mid_side_diameter)
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
                    #print( node )
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
        #print( r )
        return r

    #print( geom.root_universe )
    def do_cell( cell ):
        before = len( cmds )
        cmds.append( f"#CELL {cell.id}" )
        vol_or_body = process_node_or_fill( cell, w )
        if cell.fill_type == "material":
            cmds.append( f'group \"Material_{cell.fill.id}\" add body {{ { vol_or_body[0] } }} ' )
        after = len( cmds )
        with open( filename + f"_cell{cell.id}", "w" ) as f:
            for x in cmds[before:after]:
                f.write( x + "\n" )

    for cell in geom.root_universe._cells.values():
        if cells:
            if cell.id in cells:
                do_cell( cell )
        else:
            do_cell( cell )

    if filename:
        cmds.append( f"save as \"OPENMC_TO_CUBIT.cub\" overwrite")
        cmds.append( f"quit" )
        with open( filename, "w" ) as f:
            for x in cmds:
                f.write( x + "\n" )
            #f.write( "save sub5 \"geometry.cub\" overwrite" )

    if to_cubit:
        if cubit:
            cubit.cmd( "reset" )
            for x in cmds:
                cubit.cmd( x )

    #for x in cmds:
    #    print( x )

