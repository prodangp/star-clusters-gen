import torch
from torch import nn
import numpy as np
from utils import sigma


class DNN3L(nn.Module):
    def __init__(self, scale=16, embed_dim=8, units=100):
        super().__init__()
        self.name = f'DNN3L{units}u'
        self.nn1 = nn.Linear(7 + embed_dim, units)
        self.nn2 = nn.Linear(units, units)
        self.nn3 = nn.Linear(units, 7)
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
        self.act = nn.SiLU()

    def forward(self, x, t):
        t_proj = t[:, None] * self.W[None, :] * 2 * np.pi
        temb = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=1)
        x = torch.cat([x, temb], dim=1)
        x = self.nn1(x)
        x = self.act(x)
        x = self.nn2(x)
        x = self.act(x)
        x = self.nn3(x) / sigma(t).view(-1, 1)  # redefinition here
        return x


class DNN3L2W(nn.Module):
    def __init__(self, scale=16, embed_dim=8, units=200):
        super().__init__()
        self.name = 'DNN3L_200'
        self.nn1 = nn.Linear(7 + embed_dim, units)
        self.nn2 = nn.Linear(units, units)
        self.nn3 = nn.Linear(units, 7)
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
        self.act = nn.SiLU()

    def forward(self, x, t):
        t_proj = t[:, None] * self.W[None, :] * 2 * np.pi
        temb = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=1)
        x = torch.cat([x, temb], dim=1)
        x = self.nn1(x)
        x = self.act(x)
        x = self.nn2(x)
        x = self.act(x)
        x = self.nn3(x) / sigma(t).view(-1, 1)  # redefinition here
        return x


class DNN4L(nn.Module):
    def __init__(self, scale=16, embed_dim=8, units=400):
        super().__init__()
        self.name = 'DNN4L4W'
        self.nn1 = nn.Linear(7 + embed_dim, units)
        self.act1 = nn.SiLU()
        self.nn2 = nn.Linear(units, units)
        self.act2 = nn.SiLU()
        self.nn3 = nn.Linear(units, units)
        self.act3 = nn.SiLU()
        self.nn4 = nn.Linear(units, 7)
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x, t):
        t_proj = t[:, None] * self.W[None, :] * 2 * np.pi
        temb = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=1)
        x = torch.cat([x, temb], dim=1)
        x = self.nn1(x)
        x = self.act1(x)
        x = self.nn2(x)
        x = self.act2(x)
        x = self.nn3(x)
        x = self.act3(x)
        x = self.nn4(x) / sigma(t).view(-1, 1)  # redefinition here
        return x


class DNN6L(nn.Module):
    def __init__(self, scale=16, embed_dim=8, units=100):
        super().__init__()
        self.name = 'DNN6L'
        self.nn1 = nn.Linear(7 + embed_dim, units)
        self.act1 = nn.SiLU()
        self.nn2 = nn.Linear(units, units)
        self.act2 = nn.SiLU()
        self.nn3 = nn.Linear(units, 2*units)
        self.act3 = nn.SiLU()
        self.nn4 = nn.Linear(2*units, 2*units)
        self.act4 = nn.SiLU()
        self.nn5 = nn.Linear(2*units, units)
        self.act5 = nn.SiLU()
        self.nn6 = nn.Linear(units, 7)
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x, t):
        t_proj = t[:, None] * self.W[None, :] * 2 * np.pi
        temb = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=1)
        x = torch.cat([x, temb], dim=1)
        x = self.nn1(x)
        x = self.act1(x)
        x = self.nn2(x)
        x = self.act2(x)
        x = self.nn3(x)
        x = self.act3(x)
        x = self.nn4(x)
        x = self.act4(x)
        x = self.nn5(x)
        x = self.act5(x)
        x = self.nn6(x) / sigma(t).view(-1, 1)  # redefinition here
        return x