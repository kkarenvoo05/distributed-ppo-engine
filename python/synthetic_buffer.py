import numpy as np


def make_buffer(N_envs, T, seed=0, done_prob=0.05):
    rng = np.random.default_rng(seed)
    rewards = rng.standard_normal((N_envs, T)).astype(np.float32)
    values = rng.standard_normal((N_envs, T + 1)).astype(np.float32)
    dones = (rng.random((N_envs, T)) < done_prob).astype(np.float32)
    return rewards, values, dones

