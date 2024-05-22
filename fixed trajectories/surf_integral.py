# /********************************************************************************
#  *   Copyright (C) 2023 by Oskar Wojdeł                                         *
#  *                                                                              *
#  *   This program is free software; you can redistribute it and/or modify       *
#  *   it under the terms of the GNU General Public License as published by       *
#  *   the Free Software Foundation; either version 3 of the License, or          *
#  *   (at your option) any later version.                                        *
#  *                                                                              *
#  *   This program is distributed in the hope that it will be useful,            *
#  *   but WITHOUT ANY WARRANTY; without even the implied warranty of             *
#  *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
#  *   GNU General Public License for more details.                               *
#  *                                                                              *
#  *   License terms: https://www.gnu.org/licenses/gpl-3.0.txt                    *
#  *                                                                              *
#  ********************************************************************************/

import numpy as np
from math import pi
from numba import njit, guvectorize, float64, int64


def scalar_func(point, vertex1, vertex2, vertex3, tol=1e-10):
    '''Integral of the 1/r kernel over a triangle.

    This function calculates the integral of 1/norm(x - x0) for some
    given x0, and x in a triangle given by 3 vertices.
    :param point: np.array of shape (3,). x0 at which to calculate
    :param vertex1: shape = (3,). First vertex of triangle
    :param vertex2: shape = (3,). Second vertex of triangle
    :param vertex3: shape = (3,). Third vertex of triangle
    :param tol: float used internally for some calculations
    :return: float, value of the integral
    '''
    u21 = vertex2 - vertex1
    u21 = u21 / np.linalg.norm(u21)
    D1 = np.dot(u21, vertex3 - vertex1)
    D2 = np.dot(u21, vertex3 - vertex2)
    D = [D1, D2]

    if (D1 == 0) or (D2 == 0):
        return scalar_func(point, vertex2, vertex3, vertex1, tol)

    # A1 = np.sum(np.square(vertex3 - vertex1))
    # A2 = np.sum(np.square(vertex3 - vertex2))
    A1 = np.dot(vertex3 - vertex1, vertex3 - vertex1)
    A2 = np.dot(vertex3 - vertex2, vertex3 - vertex2)
    A = [A1, A2]

    B1 = - 2 * np.dot(vertex3 - vertex1, point - vertex1)
    B2 = - 2 * np.dot(vertex3 - vertex2, point - vertex2)
    B = [B1, B2]

    # C1 = np.sum(np.square(point - vertex1))
    # C2 = np.sum(np.square(point - vertex2))
    C1 = np.dot(point - vertex1, point - vertex1)
    C2 = np.dot(point - vertex2, point - vertex2)
    C = [C1, C2]

    E1 = np.dot(u21, vertex1 - point)
    E2 = np.dot(u21, vertex2 - point)
    E = [E1, E2]

    a1 = A1 / (D1**2)
    a2 = A2 / (D2**2)
    a = [a1, a2]

    b1 = (B1 * D1 - 2 * A1 * E1) / (D1**2)
    b2 = (B2 * D2 - 2 * A2 * E2) / (D2**2)
    b = [b1, b2]

    c1 = C1 - B1 * E1 / D1 + A1 * (E1**2) / (D1**2)
    c2 = C2 - B2 * E2 / D2 + A2 * (E2**2) / (D2**2)
    c = [c1, c2]

    # print(c1 / (a1 - 1) - b1**2 / (4 * (a1 - 1)**2))
    # print(c2 / (a2 - 1) - b2**2 / (4 * (a2 - 1)**2))

    if c1 / (a1 - 1) - b1**2 / (4 * (a1 - 1)**2) < 0:
        print(c1 / (a1 - 1) - b1**2 / (4 * (a1 - 1)**2))
        d1 = 0
    else:
        d1 = np.sqrt(c1 / (a1 - 1) - b1**2 / (4 * (a1 - 1)**2))

    if c2 / (a2 - 1) - b2**2 / (4 * (a2 - 1)**2) < 0:
        print(c2 / (a2 - 1) - b2**2 / (4 * (a2 - 1)**2))
        d2 = 0
    else:
        d2 = np.sqrt(c2 / (a2 - 1) - b2**2 / (4 * (a2 - 1)**2))
    d = [d1, d2]

    def func(eta):
        R1 = np.sqrt(A1 * eta**2 + B1 * eta + C1)
        R2 = np.sqrt(A2 * eta**2 + B2 * eta + C2)
        R = [R1, R2]
        x = [D1 * eta + E1, D2 * eta + E2]
        u = [x[0] + b1 / (2 * (a1 - 1)), x[1] + b2 / (2 * (a2 - 1))]
        val = [0, 0]
        for i in [0, 1]:
            if abs(u[i]) <= tol:
                val[i] = 0
            else:
                # val[i] = u[i] * np.log(R[i] + x[i])
                val[i] = u[i] * np.log(R[i] + x[i])

            if abs(b[i]) > tol:
                in_log = np.abs(2 * np.sqrt(a[i]) * R[i] + 2 * a[i] * x[i] + b[i])
                # print(in_log, b[i])
                val[i] -= b[i] / (2 * np.sqrt(a[i]) * (a[i] - 1)) * np.log(in_log)

            if abs(d[i]) > tol:
                val[i] += d[i] * np.arctan(u[i] / d[i])
                val[i] -= d[i] * np.arctan2(2 * d[i] * R[i] * (a[i] - 1), b[i] * x[i] + 2 * c[i])

        return val[1] / D2 - val[0] / D1

    return (func(1) - func(0)) * np.linalg.norm(np.cross(u21, vertex3 - vertex1)) / (4 * pi)


@njit
def vec_func(pts, v1, v2, v3, tol=1e-10):
    '''Integral of the 1/r kernel for one triangle at multiple points

    Same as scalar_func, but for multiple points simultaneously
    :param pts: shape = (N, 3). N points where to calculate the integral
    :param v1, v2, v3: shape = (3,). Vertices of triangle
    :param tol: float, tolerance (used internally)
    :return: np.array of shape (N,). Integral for each point
    '''
    u21 = v2 - v1
    u21 = u21 / np.linalg.norm(u21)     # Vector
    D1 = np.dot(u21, v3 - v1)           # Scalar
    D2 = np.dot(u21, v3 - v2)           # Scalar

    if (D1 == 0) or (D2 == 0):
        return vec_func(pts, v2, v3, v1, tol)

    A1 = np.dot(v3 - v1, v3 - v1)
    A2 = np.dot(v3 - v2, v3 - v2)

    B1 = - 2 * np.dot(pts - v1[np.newaxis, :], v3 - v1)
    B2 = - 2 * np.dot(pts - v2[np.newaxis, :], v3 - v2)

    C1 = np.sum(np.square(pts - v1[np.newaxis, :]), axis=1)
    C2 = np.sum(np.square(pts - v2[np.newaxis, :]), axis=1)

    E1 = np.dot(v1[np.newaxis, :] - pts, u21)
    E2 = np.dot(v2[np.newaxis, :] - pts, u21)

    a1 = A1 / (D1 ** 2)
    a2 = A2 / (D2 ** 2)

    b1 = (B1 * D1 - 2 * A1 * E1) / (D1 ** 2)
    b2 = (B2 * D2 - 2 * A2 * E2) / (D2 ** 2)

    c1 = C1 - B1 * E1 / D1 + A1 * (E1 * E1) / (D1 ** 2)
    c2 = C2 - B2 * E2 / D2 + A2 * (E2 * E2) / (D2 ** 2)

    a = [a1, a2]  # Scalars
    b = [b1, b2]  # Vectors
    c = [c1, c2]  # Vectors

    d1 = c1 / (a1 - 1) - b1 * b1 / (4 * (a1 - 1)**2)
    d1[d1 < tol] = 0
    d1 = np.sqrt(d1)

    d2 = c2 / (a2 - 1) - b2 * b2 / (4 * (a2 - 1)**2)
    d2[d2 < tol] = 0
    d2 = np.sqrt(d2)

    d = [d1, d2]    # Vectors

    def func(eta):
        R1 = np.sqrt(A1 * eta**2 + B1 * eta + C1)
        R2 = np.sqrt(A2 * eta**2 + B2 * eta + C2)
        R = [R1, R2]    # Vectors
        x = [D1 * eta + E1, D2 * eta + E2]  # Vectors
        u = [x[0] + b1 / (2 * (a1 - 1)), x[1] + b2 / (2 * (a2 - 1))]    # Vectors
        val = [np.zeros(R1.shape), np.zeros(R1.shape)]
        for i in [0, 1]:
            tmp = R[i] + x[i]
            tmp[tmp <= tol] = 1
            val[i] = u[i] * np.log(tmp)

            tmp = np.abs(2 * np.sqrt(a[i]) * R[i] + 2 * a[i] * x[i] + b[i])
            tmp[tmp <= tol] = 1
            val[i] -= b[i] / (2 * np.sqrt(a[i]) * (a[i] - 1)) * np.log(tmp)

            val[i] += d[i] * (np.arctan2(u[i], d[i]) - np.arctan2(2 * d[i] * R[i] * (a[i] - 1), b[i] * x[i] + 2 * c[i]))

        return val[1] / D2 - val[0] / D1

    return (func(1) - func(0)) * np.linalg.norm(np.cross(u21, v3 - v1)) / (4 * pi)


@njit(cache=True)
def mat_func(pts, vertices, triangles, charges, tol=1e-10):
    '''Proportional to potential at multiple points of multiple triangles

    Returns the sum for each triangle of the corresponding charge times
    the corresponding integral (1/r).
    :param pts: shape = (N, 3)
    :param vertices: shape = (K, 3)
    :param triangles: shape = (M, 3). Integers in the range [0, K - 1]
    :param charges: shape = (M,)
    :param tol: float
    :return: shape = (N,). Sum of charge * integral for each triangle
    '''
    out = np.zeros((pts.shape[0]))

    for tri in range(triangles.shape[0]):
        vert = triangles[tri, :]
        out += charges[tri] * vec_func(pts, vertices[vert[0], :], vertices[vert[1], :], vertices[vert[2], :], tol)
    return out


@guvectorize([(float64[:, :], float64[:], float64[:], float64[:], float64, float64[:])],
             "(n,d),(d),(d),(d),()->(n)", cache=True, nopython=True)
def guvec_func(pts, u1, u2, u3, q, out):
    '''Integral of the 1/r kernel for one triangle at multiple points

    Optimized version of vec_function. Technically, after decorating,
    the signature changes. However, due to uninitialized values, 'out'
    should ALWAYS be passed as an argument, either full of 0s, or as the
    vector used in the previous iteration, for the previous triangle.
    :param pts: shape = (N, 3). N points where to calculate the integral
    :param u1, u2, u3: shape = (3,). Vertices of triangle
    :param q: float. charge
    :param out: shape = (N,). The results will be ADDED to it
    '''
    v1, v2, v3 = u1, u2, u3
    for cycle in range(3):
        u21 = v2 - v1
        u21 = u21 / np.linalg.norm(u21)  # Vector
        D1 = np.dot(u21, v3 - v1)  # Scalar
        D2 = np.dot(u21, v3 - v2)  # Scalar

        if (D1 != 0) and (D2 != 0):
            break
        v1, v2, v3 = v2, v3, v1

    A1 = np.dot(v3 - v1, v3 - v1)
    A2 = np.dot(v3 - v2, v3 - v2)

    B1 = - 2 * np.dot(pts - v1[np.newaxis, :], v3 - v1)
    B2 = - 2 * np.dot(pts - v2[np.newaxis, :], v3 - v2)

    C1 = np.sum(np.square(pts - v1[np.newaxis, :]), axis=1)
    C2 = np.sum(np.square(pts - v2[np.newaxis, :]), axis=1)

    E1 = np.dot(v1[np.newaxis, :] - pts, u21)
    E2 = np.dot(v2[np.newaxis, :] - pts, u21)

    a1 = A1 / (D1 ** 2)
    a2 = A2 / (D2 ** 2)

    b1 = (B1 * D1 - 2 * A1 * E1) / (D1 ** 2)
    b2 = (B2 * D2 - 2 * A2 * E2) / (D2 ** 2)

    c1 = C1 - B1 * E1 / D1 + A1 * (E1 * E1) / (D1 ** 2)
    c2 = C2 - B2 * E2 / D2 + A2 * (E2 * E2) / (D2 ** 2)

    a = [a1, a2]  # Scalars
    b = [b1, b2]  # Vectors
    c = [c1, c2]  # Vectors

    d1 = c1 / (a1 - 1) - b1 * b1 / (4 * (a1 - 1) ** 2)
    d1[d1 < 1e-10] = 0
    d1 = np.sqrt(d1)

    d2 = c2 / (a2 - 1) - b2 * b2 / (4 * (a2 - 1) ** 2)
    d2[d2 < 1e-10] = 0
    d2 = np.sqrt(d2)

    d = [d1, d2]  # Vectors

    def func(eta):
        R1 = np.sqrt(A1 * eta ** 2 + B1 * eta + C1)
        R2 = np.sqrt(A2 * eta ** 2 + B2 * eta + C2)
        R = [R1, R2]  # Vectors
        x = [D1 * eta + E1, D2 * eta + E2]  # Vectors
        u = [x[0] + b1 / (2 * (a1 - 1)), x[1] + b2 / (2 * (a2 - 1))]  # Vectors
        val = [np.zeros(R1.shape), np.zeros(R1.shape)]
        for i in [0, 1]:
            tmp = R[i] + x[i]
            tmp[tmp <= 1e-10] = 1
            val[i] = u[i] * np.log(tmp)

            tmp = np.abs(2 * np.sqrt(a[i]) * R[i] + 2 * a[i] * x[i] + b[i])
            tmp[tmp <= 1e-10] = 1
            val[i] -= b[i] / (2 * np.sqrt(a[i]) * (a[i] - 1)) * np.log(tmp)

            val[i] += d[i] * (np.arctan2(u[i], d[i]) - np.arctan2(2 * d[i] * R[i] * (a[i] - 1), b[i] * x[i] + 2 * c[i]))

        return val[1] / D2 - val[0] / D1

    out += q * (func(1) - func(0)) * np.linalg.norm(np.cross(u21, v3 - v1)) / (4 * pi)


@guvectorize([(float64[:, :], float64[:, :], int64[:, :], float64[:], float64[:])],
             "(m,d),(n,d),(o,d),(o)->(m)", cache=True, nopython=True)
def gumat_func(pts, vertices, triangles, q, out):
    '''Integral of 1/r for multiple triangles at multiple points

    Optimized version of mat_func. Should be the fastest (but always
    test it on your machine!), though not by much. Perhaps aggressive
    optimization in C/C++ could make it faster, but it's probably not
    worth it.
    After decorating, the signature changes: 'out' should NOT be passed
    as an argument. Its values will be set to 0 anyways, and if the
    shapes don't match, it will throw an error. Instead, 'out' gets
    automatically allocated by numba, and in the end it is returned.
    Thus:
    out = gumat_func(pts, vertices, triangles, q)
    :param pts: shape = (N, 3)
    :param vertices: shape = (K, 3)
    :param triangles: shape = (M, 3). Integers in the range [0, K - 1]
    :param q: shape = (M,). Charges of the triangles
    :return: shape = (N,). Sum of charge * integral for each triangle
    '''
    out.fill(0)
    for tri in range(triangles.shape[0]):
        vert = triangles[tri, :]
        v1 = vertices[vert[0], :]
        v2 = vertices[vert[1], :]
        v3 = vertices[vert[2], :]
        for cycle in range(3):
            u21 = v2 - v1
            u21 = u21 / np.linalg.norm(u21)  # Vector
            D1 = np.dot(u21, v3 - v1)  # Scalar
            D2 = np.dot(u21, v3 - v2)  # Scalar

            if (D1 != 0) and (D2 != 0):
                break
            v1, v2, v3 = v2, v3, v1

        A1 = np.dot(v3 - v1, v3 - v1)
        A2 = np.dot(v3 - v2, v3 - v2)

        B1 = - 2 * np.dot(pts - v1[np.newaxis, :], v3 - v1)
        B2 = - 2 * np.dot(pts - v2[np.newaxis, :], v3 - v2)

        C1 = np.sum(np.square(pts - v1[np.newaxis, :]), axis=1)
        C2 = np.sum(np.square(pts - v2[np.newaxis, :]), axis=1)

        E1 = np.dot(v1[np.newaxis, :] - pts, u21)
        E2 = np.dot(v2[np.newaxis, :] - pts, u21)

        a1 = A1 / (D1 ** 2)
        a2 = A2 / (D2 ** 2)

        b1 = (B1 * D1 - 2 * A1 * E1) / (D1 ** 2)
        b2 = (B2 * D2 - 2 * A2 * E2) / (D2 ** 2)

        c1 = C1 - B1 * E1 / D1 + A1 * (E1 * E1) / (D1 ** 2)
        c2 = C2 - B2 * E2 / D2 + A2 * (E2 * E2) / (D2 ** 2)

        a = [a1, a2]  # Scalars
        b = [b1, b2]  # Vectors
        c = [c1, c2]  # Vectors

        d1 = c1 / (a1 - 1) - b1 * b1 / (4 * (a1 - 1) ** 2)
        d1[d1 < 1e-10] = 0
        d1 = np.sqrt(d1)

        d2 = c2 / (a2 - 1) - b2 * b2 / (4 * (a2 - 1) ** 2)
        d2[d2 < 1e-10] = 0
        d2 = np.sqrt(d2)

        d = [d1, d2]  # Vectors

        f = [np.zeros(B1.shape), np.zeros(B1.shape)]
        for eta in [0, 1]:
            R1 = np.sqrt(A1 * eta ** 2 + B1 * eta + C1)
            R2 = np.sqrt(A2 * eta ** 2 + B2 * eta + C2)
            R = [R1, R2]  # Vectors
            x = [D1 * eta + E1, D2 * eta + E2]  # Vectors
            u = [x[0] + b1 / (2 * (a1 - 1)),
                 x[1] + b2 / (2 * (a2 - 1))]  # Vectors
            val = [np.zeros(R1.shape), np.zeros(R1.shape)]
            for i in [0, 1]:
                tmp = R[i] + x[i]
                tmp[tmp <= 1e-10] = 1
                val[i] = u[i] * np.log(tmp)

                tmp = np.abs(2 * np.sqrt(a[i]) * R[i] + 2 * a[i] * x[i] + b[i])
                tmp[tmp <= 1e-10] = 1
                val[i] -= b[i] / (2 * np.sqrt(a[i]) * (a[i] - 1)) * np.log(tmp)

                val[i] += d[i] * (np.arctan2(u[i], d[i]) - np.arctan2(
                    2 * d[i] * R[i] * (a[i] - 1), b[i] * x[i] + 2 * c[i]
                ))

            f[eta] = val[1] / D2 - val[0] / D1

        out += q[tri] * (f[1] - f[0]) * np.linalg.norm(np.cross(u21, v3 - v1)) / (4 * pi)
