import torch
import torch.nn as nn
import numpy as np

class KalmanFilter(nn.Module):
    '''
    Q: transition_covariance
    R: observation_covariance
    H: Observation Matrix
    F: state transition matrix
    '''
    def __init__(self, obs_dim, state_dim=1, Q=0.01, R=1):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.obs_dim = obs_dim

        self.F = torch.eye(state_dim).to(self.device)
        self.H = torch.eye(obs_dim, state_dim).to(self.device)

        if isinstance(Q, np.ndarray): self.Q = torch.from_numpy(Q).float().to(self.device)
        else: self.Q = torch.eye(obs_dim).mul(Q).to(self.device)

        self.R = torch.eye(obs_dim).mul(R).to(self.device)

        # State Mean
        self.x = torch.zeros(state_dim).to(self.device)
        # State Covariance
        self.P = torch.ones((state_dim, state_dim)).to(self.device)
        self.counter = 0
    
    def _predict(self):
        x_new_prior = torch.matmul(self.F, self.x)
        P_new_prior = torch.matmul(self.F, torch.matmul(self.P, self.F.T)) + self.Q
        return x_new_prior, P_new_prior

    def predict(self):
        x, p = self._predict()
        return x.cpu().numpy(), p.cpu().numpy()
    
    def _update(self, x_new, P_new, z_new, h):
        # Predicted Observation Mean
        hx = torch.matmul(h, x_new)

        # Predicted observation Covariance
        hp = torch.matmul(h, torch.matmul(P_new, h.T)) + self.R

        K_gain = torch.matmul(P_new, torch.matmul(h.T, torch.linalg.pinv(hp)))
        self.x = x_new + torch.matmul(K_gain, (z_new - hx))
        self.P = P_new - torch.matmul(K_gain, torch.matmul(h, P_new))
        self.counter += 1
        return self.x.cpu().numpy(), self.P.cpu().numpy()

    def step(self, z_new, h_new=None):
        '''
        z_new: New Data Value
        h_new: Observation Matrix
        '''
        if isinstance(z_new, float): z_new = torch.tensor([[z_new]]).float().to(self.device)
        elif isinstance(z_new, np.ndarray): z_new = torch.from_numpy(z_new).float().to(self.device)
        if isinstance(h_new, np.ndarray): h_new = torch.from_numpy(h_new).float().to(self.device)

        if h_new is not None: h = h_new
        else: h = self.H

        if self.counter == 0:
            return self._update(self.x, self.P, z_new, h)
        else:
            x_new, P_new = self._predict()
            return self._update(x_new, P_new, z_new, h)

    def filter(self, arr: np.ndarray, obs=None):
        if obs is not None:
            assert len(arr) == len(obs)
            obs = torch.from_numpy(obs).float().to(self.device)
        
        if len(arr.shape) == 1: arr = arr.reshape(-1, 1)
        arr = torch.from_numpy(arr).float().to(self.device).T
        res = []

        for i in range(arr.shape[-1]):
            z = arr[:, i]
            if obs is not None: x, _ = self.step(z, obs[i])
            else: x, _ = self.step(z)
            res.append(x)
        return np.vstack(res)
