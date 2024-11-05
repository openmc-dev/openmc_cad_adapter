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
    eigen_results = np.linalg.eig(Aa)
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
        return (gq_type, A_, B_, C_, K_, translation, rotation_matrix)