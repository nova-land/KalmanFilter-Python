import torch
import torch.nn as nn
import numpy as np

class KalmanFilter(nn.Module):
    def __init__(self, n_feature, Q=0.01, R=1):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.n_feature = n_feature

        self.I = torch.eye(n_feature).to(self.device)
        self.F = torch.eye(n_feature).to(self.device)
        self.H = torch.eye(n_feature).to(self.device)

        self.Q = torch.eye(n_feature).mul(Q).to(self.device)
        self.R = torch.eye(n_feature).mul(R).to(self.device)

        # Filter State
        self.x = torch.zeros((n_feature, 1)).to(self.device)
        self.P = torch.ones((n_feature, n_feature)).to(self.device)
    
    def predict(self):
        x_new_prior = torch.matmul(self.F, self.x)
        P_new_prior = torch.matmul(torch.matmul(self.F, self.P), self.F.T) + self.Q
        return x_new_prior.numpy(), P_new_prior.numpy()

    def step(self, z_new):
        if isinstance(z_new, float): z_new = torch.tensor([[z_new]]).float().to(self.device)
        # predict
        x_new_prior = torch.matmul(self.F, self.x)
        P_new_prior = torch.matmul(torch.matmul(self.F, self.P), self.F.T) + self.Q

        # update
        K_new = torch.matmul(torch.matmul(P_new_prior, self.H.T), torch.linalg.pinv(torch.matmul(torch.matmul(self.H, P_new_prior), self.H.T) + self.R))
        self.x = x_new_prior + torch.matmul(K_new, (z_new - torch.matmul(self.H, x_new_prior)))
        self.P = torch.matmul(self.I - torch.matmul(K_new, self.H), P_new_prior)
        return self.x.cpu().numpy()

    def filter(self, arr: np.ndarray):
        arr = torch.from_numpy(arr).float().to(self.device).T
        res = []

        for i in range(arr.shape[-1]):
            z = arr[:, i].reshape(-1, 1)
            x = self.step(z)
            res.append(x)
        res = np.hstack(res).T
        return res