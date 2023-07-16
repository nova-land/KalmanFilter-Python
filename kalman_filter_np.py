import numpy as np

class KalmanFilter():
    def __init__(self, obs_dim, state_dim=1, Q=0.01, R=1):   
        '''
        Q: transition_covariance
        R: observation_covariance
        H: Observation Matrix
        F: state transition matrix
        '''
        self.H = np.eye(obs_dim, state_dim)
        self.F = np.eye(state_dim)

        if isinstance(Q, np.ndarray): self.Q = Q
        else: self.Q = np.eye(obs_dim) * Q
        self.R = np.eye(obs_dim) * R

        # State Mean
        self.x = np.zeros(state_dim)
        # State Covariance
        self.P = np.ones((state_dim,state_dim))
        self.counter = 0

    def predict(self):
        x_P_newrior = np.dot(self.F, self.x)
        P_P_newrior = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return x_P_newrior, P_P_newrior

    def update(self, x_new, P_new, z_new, h):
        # Predicted Observation Mean
        hx = np.dot(h, x_new)

        # Predicted Observation Covariance
        hp = np.dot(h, np.dot(P_new, h.T)) + self.R

        # update
        K_new = np.dot(P_new, np.dot(h.T, np.linalg.pinv(hp)))

        self.x = x_new + np.dot(K_new, (z_new - hx))
        self.P = P_new - np.dot(K_new, np.dot(h, P_new))
        self.counter += 1
        return self.x, self.P

    def step(self, z_new, h_new=None):
        if isinstance(z_new, float): z_new = np.array(z_new).reshape(-1, 1)
        if h_new is not None:
            h = h_new
        else: h = self.H

        if self.counter == 0:
            return self.update(self.x, self.P, z_new, h)
        else:
            x_new, P_new = self.predict()
            return self.update(x_new, P_new, z_new, h)

    def filter(self, arr: np.ndarray, obs: np.ndarray=None):
        if obs is not None:
            assert len(arr) == len(obs)
        if len(arr.shape) == 1: arr = arr.reshape(-1, 1)
        
        arr = arr.T
        res = []

        for i in range(arr.shape[-1]):
            z = arr[:, i]
            if obs is not None:
                x, _ = self.step(z, obs[i])
            else: x, _ = self.step(z)
            res.append(x)
        return np.vstack(res)