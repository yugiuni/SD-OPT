import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.inits import glorot


class GPFplusAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int, T1: int, group1: int, k: float):
        super(GPFplusAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.p_list_normalization = Ortho_Trans(T=T1, norm_groups=group1)
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()
        self.p_num = p_num
        self.k = k

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def cat(self, x: Tensor, x1: Tensor, text_f: Tensor, text_f1: Tensor, Ortho: bool, training=True):
        if Ortho:
            self.t = self.p_list_normalization.forward(self.p_list)
            if training:
                self.t.retain_grad()
                I = torch.eye(self.p_num).to(self.t.device)
                w = self.t @ self.t.T - I
                loss = w.pow(2).sum()
            else:
                loss = None
            output = self.add(x, x1, text_f, text_f1, self.t)
        else:
            output = self.add(x, x1, text_f, text_f1, self.p_list)
            loss = None
        return output, loss

    def add(self, x: Tensor, x1: Tensor, text_f: Tensor, text_f1: Tensor, y: Tensor):
        x1 += self.k * text_f1
        score = self.a(x1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(y)
        return x + p


class Ortho_Trans(torch.nn.Module):
    def __init__(self, T=5, norm_groups=1, *args, **kwargs):
        super(Ortho_Trans, self).__init__()
        self.T = T
        self.norm_groups = norm_groups
        self.eps = 1e-5

    def matrix_power3(self, Input):
        B = torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def forward(self, weight: torch.Tensor):
        assert weight.shape[0] % self.norm_groups == 0
        Z = weight.view(self.norm_groups, weight.shape[0] // self.norm_groups, -1)
        Zc = Z - Z.mean(dim=-1, keepdim=True)
        S = torch.matmul(Zc, Zc.transpose(1, 2))
        eye = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        S = S + self.eps * eye
        norm_S = S.norm(p='fro', dim=(1, 2), keepdim=True)
        S = S.div(norm_S)
        B = [None] * (self.T + 1)
        B[0] = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        for t in range(self.T):
            B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, self.matrix_power3(B[t]), S)
        W = B[self.T].matmul(Zc).div_(norm_S.sqrt())
        return W.view_as(weight)