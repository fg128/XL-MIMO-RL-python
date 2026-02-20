import numpy as np

from classes.config import Config


def get_channel(config: Config, loc):
    """Computes the channel vector between the antenna array and a location.

    Args:
        config: Config object.
        loc:    Location vector [x, y, z].

    Returns:
        h: Channel vector (Nt, 1) complex array.
    """
    loc = np.asarray(loc, dtype=float).reshape(3, 1)
    array_center = np.zeros((3, 1))

    # 1. Center Distance (d)
    d = np.linalg.norm(loc - array_center)

    # 2. Large Scale Fading Coefficient (beta_ub) [Friis Path Loss: (lambda / 4*pi*d)^2]
    beta = (config.lam / (4 * np.pi * d)) ** 2

    # 3. Array Response Vector (USW model) (a)
    d_vec = np.sqrt(np.sum((config.pos - loc) ** 2, axis=0))
    a_usw = np.exp(-1j * config.k * d_vec)

    # 4. Channel Vector h
    h = np.sqrt(beta) * np.exp(-1j * config.k * d) * a_usw.reshape(-1, 1)
    return h
