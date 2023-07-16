import numpy as np

class KalmanFilter():
    def __init__(self, n_feature, Q=0.01, R=1):   
        '''
        Q: transition_covariance
        R: observation_covariance
        '''     
        self.I = np.eye(n_feature)
        self.F = np.eye(n_feature)
        self.H = np.eye(n_feature)

        self.Q = np.eye(n_feature) * Q
        self.R = np.eye(n_feature) * R

        # Filter State
        self.x = np.zeros((n_feature, 1))
        self.P = np.ones((n_feature, n_feature))

    def step(self, z_new):
        if isinstance(z_new, float): z_new = np.array(z_new).reshape(-1, 1)
        # predict
        x_new_prior = np.dot(self.F, self.x)
        P_new_prior = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        # update
        K_new = np.dot(np.dot(P_new_prior, self.H.T), np.linalg.pinv(np.dot(np.dot(self.H, P_new_prior), self.H.T) + self.R))
        self.x = x_new_prior + np.dot(K_new, (z_new - np.dot(self.H, x_new_prior)))
        self.P = np.dot(self.I - np.dot(K_new, self.H), P_new_prior)
        return self.x, self.P

    def filter(self, arr: np.ndarray):
        arr = arr.T
        res = []

        for i in range(arr.shape[-1]):
            z = arr[:, i].reshape(-1, 1)
            x, _ = self.step(z)
            res.append(x)
        res = np.hstack(res).T
        return res