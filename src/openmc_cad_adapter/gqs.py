import numpy as np
from numpy.linalg import matrix_rank

UNKNOWN_QUADRIC = -1
ELLIPSOID = 1
ONE_SHEET_HYPERBOLOID = 2
TWO_SHEET_HYPERBOLOID = 3
ELLIPTIC_CONE = 4
ELLIPTIC_PARABOLOID = 5
HYPERBOLIC_PARABOLOID = 6
ELLIPTIC_CYLINDER = 7
HYPERBOLIC_CYLINDER = 8
PARABOLIC_CYLINDER = 9


def _make_right_handed(A):
    # Get the size of the matrix
    n = A.shape[0]

    # List to track used rows for the diagonal
    used_rows = set()

    # List to store the new column order
    column_order = []

    # Tolerance for ambiguity
    tolerance = 1e-8

    # Determine the best column for each diagonal position
    for i in range(n):  # For each diagonal position
        max_value = -np.inf
        best_col = -1
        ambiguous_cols = []

        for j in range(n):  # Check each column
            if j not in column_order:  # Only consider unused columns
                value = abs(A[i, j])
                if value > max_value + tolerance:
                    max_value = value
                    best_col = j
                    ambiguous_cols = []  # Clear ambiguities
                elif abs(value - max_value) < tolerance:  # Handle ambiguity
                    ambiguous_cols.append(j)

        if best_col != -1:
            column_order.append(best_col)
            used_rows.add(i)
        elif ambiguous_cols:  # Handle ambiguous columns
            ambiguous_cols.append(best_col)
            # Check for orthogonality with already selected columns
            for col1, col2 in zip(ambiguous_cols, ambiguous_cols[1:]):
                v1 = A[:, col1]
                v2 = A[:, col2]
                if abs(np.dot(v1, v2)) < tolerance:  # Prefer orthogonal vectors
                    best_col = col1 if col1 not in column_order else col2
                    column_order.append(best_col)
                    break

    # Reorder the columns of the matrix
    A_reordered = A[:, column_order]

    # Ensure right-handedness of the resulting matrix
    v1 = A_reordered[:, 0]
    v2 = A_reordered[:, 1]
    v3 = A_reordered[:, 2]

    cross_product = np.cross(v1, v2)
    if np.dot(cross_product, v3) < 0:  # Check for left-handedness
        # Swap two columns to fix handedness (e.g., swap the last two columns)
        A_reordered[:, [1, 2]] = A_reordered[:, [2, 1]]
        column_order[1], column_order[2] = column_order[2], column_order[1]

    return A_reordered, column_order


def characterize_general_quadratic( surface ): #s surface
    gq_tol = 1e-6
    equivalence_tol = 1e-8

    a = np.round(surface.coefficients['a'], 8)
    b = np.round(surface.coefficients['b'], 8)
    c = np.round(surface.coefficients['c'], 8)
    d = np.round(surface.coefficients['d'], 8)
    e = np.round(surface.coefficients['e'], 8)
    f = np.round(surface.coefficients['f'], 8)
    g = np.round(surface.coefficients['g'], 8)
    h = np.round(surface.coefficients['h'], 8)
    j = np.round(surface.coefficients['j'], 8)
    k = np.round(surface.coefficients['k'], 8)

    #coefficient matrix
    Aa = np.asarray([[a, d/2, f/2],
                     [d/2, b, e/2],
                     [f/2, e/2, c]])
    #hessian matrix
    Ac = np.asarray([[a, d/2, f/2, g/2],
                     [d/2, b, e/2, h/2],
                     [f/2, e/2, c, j/2],
                     [g/2, h/2, j/2, k]])
    rank_Aa = matrix_rank( Aa )
    rank_Ac = matrix_rank( Ac )

    det_Ac = np.linalg.det(Ac)
    if np.abs( det_Ac ) < gq_tol:
        delta = 0
    else:
        delta = -1 if det_Ac < 0 else 1

    eigenvalues, eigenvectors = np.linalg.eig(Aa)

    eigenvectors, order = _make_right_handed(eigenvectors)
    eigenvalues = eigenvalues[order] # Reorder eigenvalues

    signs = np.array([0, 0, 0])
    for i in range(0, 3):
        if eigenvalues[i] > -1 * gq_tol:
            signs[i] = 1
        else:
            signs[i] = -1

    S = 1 if np.abs( signs.sum() ) == 3 else -1

    B = np.array([[ -g/2], [-h/2], [-j/2 ]])

    Aai = np.linalg.pinv( Aa )

    C = np.dot(Aai, B)

    dx = C[0]
    dy = C[1]
    dz = C[2]

    #Update the constant using the resulting translation
    K_ = k + (g/2)*dx + (h/2)*dy + (j/2)*dz
    K_ = K_[0]

    if rank_Aa == 2 and rank_Ac == 3 and S == 1:
        delta = -1 if K_ * signs[0] < 0 else 1

    D = -1 if K_ * signs[0] else 1


    def find_type( rAa, rAc, delta, S, D ):
        quadric_type = UNKNOWN_QUADRIC
        if 3 == rAa and 4 == rAc and -1 == delta and 1 == S:
            quadric_type = ELLIPSOID
        elif 3 == rAa and 4 == rAc and 1 == delta and -1 == S:
            quadric_type = ONE_SHEET_HYPERBOLOID
        elif 3 == rAa and 4 == rAc and -1 == delta and -1 == S:
            quadric_type = TWO_SHEET_HYPERBOLOID
        elif 3 == rAa and 3 == rAc and 0 == delta and -1 == S:
            quadric_type = ELLIPTIC_CONE
        elif 2 == rAa and 4 == rAc and -1 == delta and 1 == S:
            quadric_type = ELLIPTIC_PARABOLOID
        elif 2 == rAa and 4 == rAc and 1 == delta and -1 == S:
            quadric_type = HYPERBOLIC_PARABOLOID
        elif 2 == rAa and 3 == rAc and -1 == delta and 1 == S:
            quadric_type = ELLIPTIC_CYLINDER
        elif 2 == rAa and 3 == rAc and 0 == delta and -1 == S:
            quadric_type = HYPERBOLIC_CYLINDER
        elif 1 == rAa and 3 == rAc and 0 == delta and 1 == S:
            quadric_type = PARABOLIC_CYLINDER
        else:
            quadric_type = UNKNOWN_QUADRIC

        # special case, replace delta with D
        if 2 == rAa and 3 == rAc and 1 == S and D != 0 :
            quadric_type = find_type( rAa, rAc, D, S, 0 )


        if quadric_type == UNKNOWN_QUADRIC:
            msg = f'UNKNOWN QUADRIC: rAa={rAa}, rAc={rAc}, delta={delta}, S={S}, D={D}'
            raise ValueError(msg)

        return quadric_type

    gq_type = find_type(rank_Aa, rank_Ac, delta, S, D)

    #set the translation
    translation = C

    rotation_matrix = eigenvectors

    for i in range( 0, 3 ):
        if abs(eigenvalues[i]) < gq_tol:
            eigenvalues[i] = 0

    A_ = eigenvalues[0]
    B_ = eigenvalues[1]
    C_ = eigenvalues[2];
    D_ = 0; E_ = 0; F_ = 0;
    G_ = 0; H_ = 0; J_ = 0;

    # alter type and coefficients for special cases
    # where coefficients are near-zero
    if gq_type == ONE_SHEET_HYPERBOLOID:
        if abs(K_) < equivalence_tol:
           K_ = 0
           gq_type = ELLIPTIC_CONE
    if gq_type == TWO_SHEET_HYPERBOLOID:
        if abs(K_) < equivalence_tol:
           K_ = 0
           gq_type = ELLIPTIC_CONE
    if gq_type == ELLIPSOID:
        if abs(A_) < equivalence_tol:
           A_ = 0
           gq_type = ELLIPTIC_CYLINDER
        elif abs( B_) < equivalence_tol:
           B_ = 0
           gq_type = ELLIPTIC_CYLINDER
        elif abs( C_) < equivalence_tol:
           C_ = 0
           gq_type = ELLIPTIC_CYLINDER

    return (gq_type, A_, B_, C_, K_, translation, rotation_matrix)


__all__ = ["characterize_general_quadratic",
           "ELLIPSOID",
           "ONE_SHEET_HYPERBOLOID",
           "TWO_SHEET_HYPERBOLOID",
           "ELLIPTIC_CONE",
           "ELLIPTIC_PARABOLOID",
           "HYPERBOLIC_PARABOLOID",
           "ELLIPTIC_CYLINDER",
           "HYPERBOLIC_CYLINDER",
           "PARABOLIC_CYLINDER"]
