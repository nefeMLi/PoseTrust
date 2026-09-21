"""Marginal covariances by selected inversion of the Cholesky factor,
not dense inversion of the information matrix.

This is the quantity the whole study interrogates. A pose-graph solver reports
it as *the* uncertainty of its estimate, and every downstream consumer treats
it as real -- so the first requirement is that it be computed exactly, with no
approximation of our own layered on top of the Laplace approximation already
baked into it. Any sloppiness here would be indistinguishable from the
miscalibration the experiments are trying to measure.

Why selected inversion rather than np.linalg.inv(H): the full inverse of a
sparse information matrix is dense, so forming it costs O(n^3) time and O(n^2)
memory in the number of poses, while only a tiny fraction of the entries are
ever wanted -- the per-pose diagonal blocks and a handful of pairs. The
Takahashi recursion computes exactly the entries inside the sparsity pattern
of the Cholesky factor and nothing else. The recursion below is the general
one; the sparse pattern that makes it cheap arrives with linear_solve.py in
week 2, and this dense implementation stays as the reference it is checked
against.
"""

from __future__ import annotations

import numpy as np

from posetrust.optimize.gauge import free_mask


def cholesky_factor(H: np.ndarray) -> np.ndarray:
    """Lower-triangular L with H == L @ L.T, for symmetric positive definite H.

    Raises if H is not positive definite, which for a pose graph almost always
    means the gauge was not fixed rather than anything subtler.
    """
    try:
        return np.linalg.cholesky(H)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "information matrix is not positive definite -- is the gauge fixed? "
            "An unanchored pose graph is singular by construction."
        ) from exc


def selected_inverse(L: np.ndarray) -> np.ndarray:
    """Takahashi recursion: recover Sigma = (L @ L.T)^-1 from its Cholesky factor.

    From H = L L^T we get L^T Sigma = L^-1, and since L^-1 is lower triangular
    its strictly-upper entries vanish. Reading row i of that identity,

        L_ii Sigma_ij + sum_{k>i} L_ki Sigma_kj = (1/L_ii) if i == j else 0

    which rearranges into a backward recursion for Sigma_ij in terms of entries
    with strictly larger indices.

    Order of operations matters and is easy to get wrong: at step i the whole
    lower-right block Sigma[i+1:, i+1:] must already be complete, because the
    off-diagonal entries of row i are built from it. Only once those are in
    place can the diagonal Sigma_ii be formed, since its sum runs over the
    column-i entries just produced. Folding the two into one matrix-vector
    product reads the not-yet-written column i as zeros and silently returns
    something that is not an inverse at all.
    """
    n = L.shape[0]
    sigma = np.zeros((n, n))
    for i in range(n - 1, -1, -1):
        below = L[i + 1 :, i]
        if below.size:
            off = -(below @ sigma[i + 1 :, i + 1 :]) / L[i, i]
            sigma[i, i + 1 :] = off
            sigma[i + 1 :, i] = off
            sigma[i, i] = (1.0 / L[i, i] - below @ sigma[i + 1 :, i]) / L[i, i]
        else:
            sigma[i, i] = 1.0 / L[i, i] ** 2
    return sigma


def covariance_matrix(
    information: np.ndarray, anchor: int, dof: int
) -> np.ndarray:
    """Full state covariance, with the anchored pose's rows and columns zero.

    The zeros are not padding: the anchored pose *is* the reference frame, so
    it has no uncertainty relative to itself. Testing for exact zeros there is
    how gauge handling is verified.
    """
    n_poses = information.shape[0] // dof
    free = free_mask(n_poses, dof, anchor)
    L = cholesky_factor(information[np.ix_(free, free)])
    sigma = np.zeros_like(information)
    sigma[np.ix_(free, free)] = selected_inverse(L)
    return sigma


def marginal_covariances(
    information: np.ndarray, anchor: int, dof: int
) -> list[np.ndarray]:
    """Per-pose marginal covariance blocks, in pose order.

    These are what a SLAM system reports and what an operator display draws as
    ellipsoids. The anchored pose's block is exactly zero.
    """
    sigma = covariance_matrix(information, anchor, dof)
    n_poses = information.shape[0] // dof
    return [sigma[k * dof : (k + 1) * dof, k * dof : (k + 1) * dof] for k in range(n_poses)]


def joint_covariance(
    information: np.ndarray, i: int, j: int, anchor: int, dof: int
) -> np.ndarray:
    """Joint covariance of poses i and j, including the cross-covariance block.

    The cross term is what makes relative uncertainty smaller than the sum of
    two absolute uncertainties: errors shared through the anchor are common
    mode and cancel. Ignoring it is a standard way to end up reporting a
    relative covariance that is far too large.
    """
    sigma = covariance_matrix(information, anchor, dof)
    idx = np.r_[i * dof : (i + 1) * dof, j * dof : (j + 1) * dof]
    return sigma[np.ix_(idx, idx)]


def relative_covariance(
    lie,
    information: np.ndarray,
    poses: list[np.ndarray],
    i: int,
    j: int,
    anchor: int,
) -> np.ndarray:
    """Covariance of the relative pose Ti^-1 @ Tj -- the gauge-invariant quantity.

    Absolute marginals depend on which pose is anchored, so they are not by
    themselves meaningful; relative ones do not, and must come out identical
    for any choice of anchor. That invariance is the sharpest available test
    that gauge handling has not quietly corrupted the covariances.

    Perturbing both poses on the right, the relative pose moves by
    -Adj(M^-1) delta_i + delta_j with M = Ti^-1 @ Tj, so the joint covariance
    propagates through J = [-Adj(M^-1), I].
    """
    dof = lie.DOF
    m_inv = lie.compose(lie.inverse(poses[j]), poses[i])
    J = np.hstack([-lie.adjoint(m_inv), np.eye(dof)])
    return J @ joint_covariance(information, i, j, anchor, dof) @ J.T
