#-*- coding:utf-8 -*-
# scipy 版本 0.17.1
from __future__ import division
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PPoly
from scipy.linalg import solve_banded, solve
from scipy._lib.six import string_types


class CubicSpline(PPoly):
    """Cubic spline data interpolator.
    Interpolate data with a piecewise cubic polynomial which is twice
    continuously differentiable [1]_. The result is represented as a `PPoly`
    instance with breakpoints matching the given data.
    Parameters
    ----------
    x : array_like, shape (n,)
        1-d array containing values of the independent variable.
        Values must be real, finite and in strictly increasing order.
    y : array_like
        Array containing values of the dependent variable. It can have
        arbitrary number of dimensions, but the length along `axis` (see below)
        must match the length of `x`. Values must be finite.
    axis : int, optional
        Axis along which `y` is assumed to be varying. Meaning that for
        ``x[i]`` the corresponding values are ``np.take(y, i, axis=axis)``.
        Default is 0.
    bc_type :
        【边界函数】
        string or 2-tuple, optional
        Boundary condition type. Two additional equations, given by the
        boundary conditions, are required to determine all coefficients of
        polynomials on each segment [2]_.
        If `bc_type` is a string, then the specified condition will be applied
        at both ends of a spline. Available conditions are:
        * 'not-a-knot' (default) 【默认】: The first and second segment at a curve end
          are the same polynomial. It is a good default when there is no
          information on boundary conditions.
        * 'periodic'【周期性】: The interpolated functions is assumed to be periodic
          of period ``x[-1] - x[0]``. The first and last value of `y` must be
          identical: ``y[0] == y[-1]``. This boundary condition will result in
          ``y'[0] == y'[-1]`` and ``y''[0] == y''[-1]``.
        * 'clamped'【收敛】: The first derivative at curves ends are zero. Assuming
          a 1D `y`, ``bc_type=((1, 0.0), (1, 0.0))`` is the same condition.
        * 'natural'【自然-课本】:
            【讲解】bc_type=((2, 0.0), (2, 0.0))后面是两个参数，前面是x0这个点，后面是xn这个点，
            (2, 0.0)中：第一个为第几阶导数，第二个表示该导数数值。2表示二阶导，0.0表示二阶导为0。
                        1表示一阶导。详细导数设置，参见，样条的两种边界条件。
            The second derivative at curve ends are zero. Assuming
          a 1D `y`, ``bc_type=((2, 0.0), (2, 0.0))`` is the same condition.
        If `bc_type` is a 2-tuple, the first and the second value will be
        applied at the curve start and end respectively. The tuple values can
        be one of the previously mentioned strings (except 'periodic') or a
        tuple `(order, deriv_values)` allowing to specify arbitrary
        derivatives at curve ends:
        * `order`: the derivative order, 1 or 2.
        * `deriv_value`: array_like containing derivative values, shape must
          be the same as `y`, excluding `axis` dimension. For example, if `y`
          is 1D, then `deriv_value` must be a scalar. If `y` is 3D with the
          shape (n0, n1, n2) and axis=2, then `deriv_value` must be 2D
          and have the shape (n0, n1).
    extrapolate : {bool, 'periodic', None}, optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. If None (default), `extrapolate` is
        set to 'periodic' for ``bc_type='periodic'`` and to True otherwise.
    Attributes
    ----------
    x : ndarray, shape (n,)
        Breakpoints. The same `x` which was passed to the constructor.
    c : ndarray, shape (4, n-1, ...)
        Coefficients of the polynomials on each segment. The trailing
        dimensions match the dimensions of `y`, excluding `axis`. For example,
        if `y` is 1-d, then ``c[k, i]`` is a coefficient for
        ``(x-x[i])**(3-k)`` on the segment between ``x[i]`` and ``x[i+1]``.
    axis : int
        Interpolation axis. The same `axis` which was passed to the
        constructor.
    Methods
    -------
    __call__
    derivative
    antiderivative
    integrate
    roots
    See Also
    --------
    Akima1DInterpolator
    PchipInterpolator
    PPoly
    Notes
    -----
    Parameters `bc_type` and `interpolate` work independently, i.e. the former
    controls only construction of a spline, and the latter only evaluation.
    When a boundary condition is 'not-a-knot' and n = 2, it is replaced by
    a condition that the first derivative is equal to the linear interpolant
    slope. When both boundary conditions are 'not-a-knot' and n = 3, the
    solution is sought as a parabola passing through given points.
    When 'not-a-knot' boundary conditions is applied to both ends, the
    resulting spline will be the same as returned by `splrep` (with ``s=0``)
    and `InterpolatedUnivariateSpline`, but these two methods use a
    representation in B-spline basis.
    .. versionadded:: 0.18.0
    Examples
    --------
    In this example the cubic spline is used to interpolate a sampled sinusoid.
    You can see that the spline continuity property holds for the first and
    second derivatives and violates only for the third derivative.
    # >>> from scipy.interpolate import CubicSpline
    # >>> import matplotlib.pyplot as plt
    # >>> x = np.arange(10)
    # >>> y = np.sin(x)
    # >>> cs = CubicSpline(x, y)
    # >>> xs = np.arange(-0.5, 9.6, 0.1)
    # >>> plt.figure(figsize=(6.5, 4))
    # >>> plt.plot(x, y, 'o', label='data')
    # >>> plt.plot(xs, np.sin(xs), label='true')
    # >>> plt.plot(xs, cs(xs), label="S")
    # >>> plt.plot(xs, cs(xs, 1), label="S'")
    # >>> plt.plot(xs, cs(xs, 2), label="S''")
    # >>> plt.plot(xs, cs(xs, 3), label="S'''")
    # >>> plt.xlim(-0.5, 9.5)
    # >>> plt.legend(loc='lower left', ncol=2)
    # >>> plt.show()
    In the second example, the unit circle is interpolated with a spline. A
    periodic boundary condition is used. You can see that the first derivative
    values, ds/dx=0, ds/dy=1 at the periodic point (1, 0) are correctly
    computed. Note that a circle cannot be exactly represented by a cubic
    spline. To increase precision, more breakpoints would be required.
    # >>> theta = 2 * np.pi * np.linspace(0, 1, 5)
    # >>> y = np.c_[np.cos(theta), np.sin(theta)]
    # >>> cs = CubicSpline(theta, y, bc_type='periodic')
    # >>> print("ds/dx={:.1f} ds/dy={:.1f}".format(cs(0, 1)[0], cs(0, 1)[1]))
    ds/dx=0.0 ds/dy=1.0
    # >>> xs = 2 * np.pi * np.linspace(0, 1, 100)
    # >>> plt.figure(figsize=(6.5, 4))
    # >>> plt.plot(y[:, 0], y[:, 1], 'o', label='data')
    # >>> plt.plot(np.cos(xs), np.sin(xs), label='true')
    # >>> plt.plot(cs(xs)[:, 0], cs(xs)[:, 1], label='spline')
    # >>> plt.axes().set_aspect('equal')
    # >>> plt.legend(loc='center')
    # >>> plt.show()
    The third example is the interpolation of a polynomial y = x**3 on the
    interval 0 <= x<= 1. A cubic spline can represent this function exactly.
    To achieve that we need to specify values and first derivatives at
    endpoints of the interval. Note that y' = 3 * x**2 and thus y'(0) = 0 and
    y'(1) = 3.
    # >>> cs = CubicSpline([0, 1], [0, 1], bc_type=((1, 0), (1, 3)))
    # >>> x = np.linspace(0, 1)
    # >>> np.allclose(x**3, cs(x))
    True
    References
    ----------
    .. [1] `Cubic Spline Interpolation
            <https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation>`_
            on Wikiversity.
    .. [2] Carl de Boor, "A Practical Guide to Splines", Springer-Verlag, 1978.
    """
    def __init__(self, x, y, axis=0, bc_type='not-a-knot', extrapolate=None):
        x, y = map(np.asarray, (x, y))

        if np.issubdtype(x.dtype, np.complexfloating):
            raise ValueError("`x` must contain real values.")

        if np.issubdtype(y.dtype, np.complexfloating):
            dtype = complex
        else:
            dtype = float
        y = y.astype(dtype, copy=False)

        axis = axis % y.ndim
        if x.ndim != 1:
            raise ValueError("`x` must be 1-dimensional.")
        if x.shape[0] < 2:
            raise ValueError("`x` must contain at least 2 elements.")
        if x.shape[0] != y.shape[axis]:
            raise ValueError("The length of `y` along `axis`={0} doesn't "
                             "match the length of `x`".format(axis))

        if not np.all(np.isfinite(x)):
            raise ValueError("`x` must contain only finite values.")
        if not np.all(np.isfinite(y)):
            raise ValueError("`y` must contain only finite values.")

        dx = np.diff(x)
        if np.any(dx <= 0):
            raise ValueError("`x` must be strictly increasing sequence.")

        n = x.shape[0]
        y = np.rollaxis(y, axis)

        bc, y = self._validate_bc(bc_type, y, y.shape[1:], axis)

        if extrapolate is None:
            if bc[0] == 'periodic':
                extrapolate = 'periodic'
            else:
                extrapolate = True

        dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
        slope = np.diff(y, axis=0) / dxr

        # If bc is 'not-a-knot' this change is just a convention.
        # If bc is 'periodic' then we already checked that y[0] == y[-1],
        # and the spline is just a constant, we handle this case in the same
        # way by setting the first derivatives to slope, which is 0.
        if n == 2:
            if bc[0] in ['not-a-knot', 'periodic']:
                bc[0] = (1, slope[0])
            if bc[1] in ['not-a-knot', 'periodic']:
                bc[1] = (1, slope[0])

        # This is a very special case, when both conditions are 'not-a-knot'
        # and n == 3. In this case 'not-a-knot' can't be handled regularly
        # as the both conditions are identical. We handle this case by
        # constructing a parabola passing through given points.
        if n == 3 and bc[0] == 'not-a-knot' and bc[1] == 'not-a-knot':
            A = np.zeros((3, 3))  # This is a standard matrix.
            b = np.empty((3,) + y.shape[1:], dtype=y.dtype)

            A[0, 0] = 1
            A[0, 1] = 1
            A[1, 0] = dx[1]
            A[1, 1] = 2 * (dx[0] + dx[1])
            A[1, 2] = dx[0]
            A[2, 1] = 1
            A[2, 2] = 1

            b[0] = 2 * slope[0]
            b[1] = 3 * (dxr[0] * slope[1] + dxr[1] * slope[0])
            b[2] = 2 * slope[1]

            s = solve(A, b, overwrite_a=True, overwrite_b=True,
                      check_finite=False)
        else:
            # Find derivative values at each x[i] by solving a tridiagonal
            # system.
            A = np.zeros((3, n))  # This is a banded matrix representation.
            b = np.empty((n,) + y.shape[1:], dtype=y.dtype)

            # Filling the system for i=1..n-2
            #                         (x[i-1] - x[i]) * s[i-1] +\
            # 2 * ((x[i] - x[i-1]) + (x[i+1] - x[i])) * s[i]   +\
            #                         (x[i] - x[i-1]) * s[i+1] =\
            #       3 * ((x[i+1] - x[i])*(y[i] - y[i-1])/(x[i] - x[i-1]) +\
            #           (x[i] - x[i-1])*(y[i+1] - y[i])/(x[i+1] - x[i]))

            A[1, 1:-1] = 2 * (dx[:-1] + dx[1:])  # The diagonal
            A[0, 2:] = dx[:-1]                   # The upper diagonal
            A[-1, :-2] = dx[1:]                  # The lower diagonal

            b[1:-1] = 3 * (dxr[1:] * slope[:-1] + dxr[:-1] * slope[1:])

            bc_start, bc_end = bc

            if bc_start == 'periodic':
                # Due to the periodicity, and because y[-1] = y[0], the linear
                # system has (n-1) unknowns/equations instead of n:
                A = A[:, 0:-1]
                A[1, 0] = 2 * (dx[-1] + dx[0])
                A[0, 1] = dx[-1]

                b = b[:-1]

                # Also, due to the periodicity, the system is not tri-diagonal.
                # We need to compute a "condensed" matrix of shape (n-2, n-2).
                # See http://www.cfm.brown.edu/people/gk/chap6/node14.html for
                # more explanations.
                # The condensed matrix is obtained by removing the last column
                # and last row of the (n-1, n-1) system matrix. The removed
                # values are saved in scalar variables with the (n-1, n-1)
                # system matrix indices forming their names:
                a_m1_0 = dx[-2]  # lower left corner value: A[-1, 0]
                a_m1_m2 = dx[-1]
                a_m1_m1 = 2 * (dx[-1] + dx[-2])
                a_m2_m1 = dx[-2]
                a_0_m1 = dx[0]

                b[0] = 3 * (dxr[0] * slope[-1] + dxr[-1] * slope[0])
                b[-1] = 3 * (dxr[-1] * slope[-2] + dxr[-2] * slope[-1])

                Ac = A[:, :-1]
                b1 = b[:-1]
                b2 = np.zeros_like(b1)
                b2[0] = -a_0_m1
                b2[-1] = -a_m2_m1

                # s1 and s2 are the solutions of (n-2, n-2) system
                s1 = solve_banded((1, 1), Ac, b1, overwrite_ab=False,
                                  overwrite_b=False, check_finite=False)

                s2 = solve_banded((1, 1), Ac, b2, overwrite_ab=False,
                                  overwrite_b=False, check_finite=False)

                # computing the s[n-2] solution:
                s_m1 = ((b[-1] - a_m1_0 * s1[0] - a_m1_m2 * s1[-1]) /
                        (a_m1_m1 + a_m1_0 * s2[0] + a_m1_m2 * s2[-1]))

                # s is the solution of the (n, n) system:
                s = np.empty((n,) + y.shape[1:], dtype=y.dtype)
                s[:-2] = s1 + s_m1 * s2
                s[-2] = s_m1
                s[-1] = s[0]
            else:
                if bc_start == 'not-a-knot':
                    A[1, 0] = dx[1]
                    A[0, 1] = x[2] - x[0]
                    d = x[2] - x[0]
                    b[0] = ((dxr[0] + 2*d) * dxr[1] * slope[0] +
                            dxr[0]**2 * slope[1]) / d
                elif bc_start[0] == 1:
                    A[1, 0] = 1
                    A[0, 1] = 0
                    b[0] = bc_start[1]
                elif bc_start[0] == 2:
                    A[1, 0] = 2 * dx[0]
                    A[0, 1] = dx[0]
                    b[0] = -0.5 * bc_start[1] * dx[0]**2 + 3 * (y[1] - y[0])

                if bc_end == 'not-a-knot':
                    A[1, -1] = dx[-2]
                    A[-1, -2] = x[-1] - x[-3]
                    d = x[-1] - x[-3]
                    b[-1] = ((dxr[-1]**2*slope[-2] +
                             (2*d + dxr[-1])*dxr[-2]*slope[-1]) / d)
                elif bc_end[0] == 1:
                    A[1, -1] = 1
                    A[-1, -2] = 0
                    b[-1] = bc_end[1]
                elif bc_end[0] == 2:
                    A[1, -1] = 2 * dx[-1]
                    A[-1, -2] = dx[-1]
                    b[-1] = 0.5 * bc_end[1] * dx[-1]**2 + 3 * (y[-1] - y[-2])

                s = solve_banded((1, 1), A, b, overwrite_ab=True,
                                 overwrite_b=True, check_finite=False)

        # Compute coefficients in PPoly form.
        t = (s[:-1] + s[1:] - 2 * slope) / dxr
        c = np.empty((4, n - 1) + y.shape[1:], dtype=t.dtype)
        c[0] = t / dxr
        c[1] = (slope - s[:-1]) / dxr - t
        c[2] = s[:-1]
        c[3] = y[:-1]

        super(CubicSpline, self).__init__(c, x, extrapolate=extrapolate)
        self.axis = axis

    @staticmethod
    def _validate_bc(bc_type, y, expected_deriv_shape, axis):
        """Validate and prepare boundary conditions.
        Returns
        -------
        validated_bc : 2-tuple
            Boundary conditions for a curve start and end.
        y : ndarray
            y casted to complex dtype if one of the boundary conditions has
            complex dtype.
        """
        if isinstance(bc_type, string_types):
            if bc_type == 'periodic':
                if not np.allclose(y[0], y[-1], rtol=1e-15, atol=1e-15):
                    raise ValueError(
                        "The first and last `y` point along axis {} must "
                        "be identical (within machine precision) when "
                        "bc_type='periodic'.".format(axis))

            bc_type = (bc_type, bc_type)

        else:
            if len(bc_type) != 2:
                raise ValueError("`bc_type` must contain 2 elements to "
                                 "specify start and end conditions.")

            if 'periodic' in bc_type:
                raise ValueError("'periodic' `bc_type` is defined for both "
                                 "curve ends and cannot be used with other "
                                 "boundary conditions.")

        validated_bc = []
        for bc in bc_type:
            if isinstance(bc, string_types):
                if bc == 'clamped':
                    validated_bc.append((1, np.zeros(expected_deriv_shape)))
                elif bc == 'natural':
                    validated_bc.append((2, np.zeros(expected_deriv_shape)))
                elif bc in ['not-a-knot', 'periodic']:
                    validated_bc.append(bc)
                else:
                    raise ValueError("bc_type={} is not allowed.".format(bc))
            else:
                try:
                    deriv_order, deriv_value = bc
                except Exception:
                    raise ValueError("A specified derivative value must be "
                                     "given in the form (order, value).")

                if deriv_order not in [1, 2]:
                    raise ValueError("The specified derivative order must "
                                     "be 1 or 2.")

                deriv_value = np.asarray(deriv_value)
                if deriv_value.shape != expected_deriv_shape:
                    raise ValueError(
                        "`deriv_value` shape {} is not the expected one {}."
                        .format(deriv_value.shape, expected_deriv_shape))

                if np.issubdtype(deriv_value.dtype, np.complexfloating):
                    y = y.astype(complex, copy=False)

                validated_bc.append((deriv_order, deriv_value))

        return validated_bc, y

