import numpy as np


def gae_reference(rewards, values, dones, gamma, lam):
    """
    rewards: [N_envs, T]
    values:  [N_envs, T+1]
    dones:   [N_envs, T]
    """
    rewards64 = rewards.astype(np.float64, copy=False)
    values64 = values.astype(np.float64, copy=False)
    dones64 = dones.astype(np.float64, copy=False)
    N, T = rewards.shape
    adv = np.zeros((N, T), dtype=np.float64)
    gae = np.zeros(N, dtype=np.float64)
    for t in reversed(range(T)):
        mask = 1.0 - dones64[:, t]
        delta = rewards64[:, t] + gamma * values64[:, t + 1] * mask - values64[:, t]
        gae = delta + gamma * lam * mask * gae
        adv[:, t] = gae
    ret = adv + values64[:, :T]
    return adv.astype(np.float32), ret.astype(np.float32)


if __name__ == "__main__":
    rewards = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 0.0, -0.5, 1.0]], dtype=np.float32)
    values = np.zeros((2, 5), dtype=np.float32)
    dones = np.array([[0, 0, 0, 1], [0, 1, 0, 0]], dtype=np.float32)
    adv, ret = gae_reference(rewards, values, dones, 0.99, 0.95)
    print("advantages")
    print(adv)
    print("returns")
    print(ret)

