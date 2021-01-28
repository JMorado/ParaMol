import numpy as np


def calculate_distance(p0, p1):
    """
    Method that calculates the distance between two points.

    Parameters
    ----------
    p0 : list of np.array
        Point 0 coordinates.
    p1 : list of np.array
        Point 1 coordinates.

    Returns
    -------
    norm : float
        Distance.
    """

    return np.linalg.norm(p1-p0)


def unit_vector(v):
    """
    Method that returns the unit vector of the vector.

    Parameters
    ----------
    v : list of np.array
        Vector to normalize.


    Returns
    -------
    unit_vector : np.array
        Unit vector
    """

    return v / np.linalg.norm(v)


def calculate_angle(v1, v2):
    """
    Method that returns the angle in radians between vectors 'v1' and 'v2'.

    Parameters
    ----------
    v1 : list of np.array
        Vector 1.
    v2 : list of np.array
        Vector 2.

    Returns
    -------
    angle_rad : float
        The angle in radians between vectors 'v1' and 'v2'.
    """

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def calculate_dihedral(p0, p1, p2, p3):
    """
    Method that returns the dihedral angle formed by 4 points using the praxeolitic formula.

    Parameters
    ----------
    p0 : list of np.array
        Point 0 coordinates.
    p1 : list of np.array
        Point 1 coordinates.
    p2 : list of np.array
        Point 2 coordinates.
    p3 : list of np.array
        Point 3 coordinates.

    Returns
    -------
    angle_rad : float
        The dihedral angle in radians.
    """

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    return np.arctan2(y, x)