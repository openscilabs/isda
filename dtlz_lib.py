# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import math

def dtlz2(n_vars, n_objs, k=None):
    """
    Generates a sample from DTLZ2.
    Intrinsic dimensionality = n_objs - 1 (Surface is hypersphere part).
    
    Args:
        n_vars: Total decision variables.
        n_objs: Number of objectives.
        k: Number of distance-related variables (g function). Default = n_vars - n_objs + 1.
    Returns:
        F: (1, n_objs) array (single sample) -> NO, we need N samples.
        REFACTOR: This function should return N samples.
    """
    raise NotImplementedError("Use generate_dtlz2 instead")

def generate_dtlz2(N=1000, M=3, n_vars=12):
    """
    Generates N samples of DTLZ2 with M objectives.
    DTLZ2 Geometry: Spherical front.
    Intrinsic Dimension: M-1.
    """
    rng = np.random.default_rng()
    
    # k = n_vars - M + 1 (usually 10)
    k = n_vars - M + 1
    
    # Generate X
    # x_position (M-1)
    # x_distance (k)
    X = rng.uniform(0.0, 1.0, size=(N, n_vars))
    
    xm = X[:, (M-1):] # Distance variables
    g = np.sum((xm - 0.5)**2, axis=1) # DTLZ2 g(x)
    
    F = np.zeros((N, M))
    
    for i in range(M):
        f = (1.0 + g)
        for j in range(M - 1 - i):
            f *= np.cos(X[:, j] * math.pi / 2.0)
        if i > 0:
            f *= np.sin(X[:, M - 1 - i] * math.pi / 2.0)
        F[:, i] = f
        
    return F, X

def generate_dtlz5(N=1000, M=3, n_vars=12):
    """
    Generates N samples of DTLZ5 (Degenerate curve).
    DTLZ5 Geometry: Curve on the sphere.
    Intrinsic Dimension: M-1 (IF I=M), but usually I < M.
    Original paper: I = number of true objectives.
    For M=3, DTLZ5 typically degenerates to a 1D curve (some papers say 2D surface is reduced).
    Standard DTLZ5(I, M):
    theta_i depends on g for i > I-1.
    """
    rng = np.random.default_rng()
    k = n_vars - M + 1
    X = rng.uniform(0.0, 1.0, size=(N, n_vars))
    
    xm = X[:, (M-1):]
    g = np.sum((xm - 0.5)**2, axis=1)
    
    # theta calculation
    # theta_i = x_i * pi / 2 for i = 0..I-2
    # theta_i = (1+2g_x)*x_i... tricky.
    
    # Standard DTLZ5 implementation
    # Let "I" be the true dimensionality parameter. usually I=2 for curve.
    
    theta = np.zeros((N, M-1))
    
    # t = g(xm)
    # theta_0 = x_0 * pi/2
    # others: pi/(4(1+g)) * (1 + 2*g*x_i)
    
    # Assuming degenerate curve M=3, True Dim = 1 (Curve) ??
    # Actually DTLZ5(2, 3) is a curve.
    
    theta[:, 0] = X[:, 0] * math.pi / 2.0
    
    gr = g[:, np.newaxis]
    for i in range(1, M-1):
        theta[:, i] = ((math.pi / (4.0 * (1.0 + gr))) * (1.0 + 2.0 * gr * X[:, i][:, np.newaxis])).ravel()
        
    F = np.zeros((N, M))
    for i in range(M):
        f = (1.0 + g)
        for j in range(M - 1 - i):
            f *= np.cos(theta[:, j])
        if i > 0:
            f *= np.sin(theta[:, M - 1 - i])
        F[:, i] = f
        
    return F, X

