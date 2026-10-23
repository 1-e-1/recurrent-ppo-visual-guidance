"""Deterministic task-nullspace restart proposals, never motion certificates."""
import numpy as np


def nullspace_starts(arm, q, *, axis=None, rotation=None, vertical_row=None,
                     max_delta=.30, count=16):
    q = np.asarray(q, float)
    if (q.shape != (8,) or not np.isfinite(q).all() or np.any(abs(q) > np.pi)
            or not np.isfinite(max_delta) or not 0 < max_delta <= np.pi
            or not isinstance(count, int) or not 1 <= count <= 32):
        raise ValueError('bounded finite restart configuration required')

    def task(x):
        t = arm.tcp(x)
        values = [12 * t[:3, 3]]
        if axis is not None:
            values.append(2 * t[:3, 2])
        if rotation is not None:
            values.append(2 * t[:3, :3].ravel())
        if vertical_row is not None:
            values.append(2 * t[2, :3])
        return np.concatenate(values)

    eps = 1e-5
    J = np.column_stack([(task(q + eps * v) - task(q - eps * v)) / (2 * eps)
                         for v in np.eye(8)])
    _, singular, V = np.linalg.svd(J, full_matrices=True)
    rank = int(np.sum(singular > max(1e-7, singular[0] * 1e-5)))
    basis = V[rank:].T
    if basis.shape[1] == 0:
        return np.empty((0, 8))
    lower = np.maximum(-np.pi, q - max_delta)
    upper = np.minimum(np.pi, q + max_delta)
    points = []
    # Signed basis directions, then deterministic mixtures; all are merely
    # nonlinear solver starts. The trust region remains centred at original q.
    for index in range(count):
        pair = index // 2
        if pair < basis.shape[1]:
            d = basis[:, pair].copy()
        else:
            d = basis @ np.cos((pair + 1) * np.arange(1, basis.shape[1] + 1))
        d *= (1 if index % 2 == 0 else -1)
        d /= max(float(np.max(abs(d))), 1e-12)
        points.append(np.clip(q + .85 * max_delta * d, lower, upper))
    return np.asarray(points)
