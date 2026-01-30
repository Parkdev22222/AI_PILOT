import math
import torch as T
import torch.nn as nn
import torch.nn.functional as F



class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, bn=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.bn1 = nn.BatchNorm1d(dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.bn2 = nn.BatchNorm1d(dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.bn3 = nn.BatchNorm1d(dim_V)
        if bn:
            self.ln0 = nn.BatchNorm1d(dim_V)
            self.ln1 = nn.BatchNorm1d(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        # Q = self.bn1(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        # K, V = self.bn2(K), self.bn3(V)

        dim_split = self.dim_V // self.num_heads
        Q_ = T.cat(Q.split(dim_split, 2), 0)
        K_ = T.cat(K.split(dim_split, 2), 0)
        V_ = T.cat(V.split(dim_split, 2), 0)

        A = T.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = T.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, bn=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, bn=bn)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, bn=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(T.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, bn=bn)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, bn=bn)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, bn=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(T.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, bn=bn)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = T.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = T.arange(0, max_len).float().unsqueeze(1)
        div_term = (T.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = T.sin(position * div_term)
        pe[:, 1::2] = T.cos(position * div_term)[:, :d_model//2]

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, device):
        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=input_dim, dim_out=128, num_heads=4),
            SAB(dim_in=128, dim_out=128, num_heads=4),
            # SAB(dim_in=64, dim_out=64, num_heads=4),
        )
        self.to(device)

    def forward(self, trajectory):
        # pos = self.position(x)
        # print(pos, x)
        x = self.enc(trajectory)
        # print(x.shape)
        return x


class TrajectoryDecoder(nn.Module):
    def __init__(self, obs_act_dim, device):
        super().__init__()
        self.dec = nn.Sequential(
            SAB(dim_in=128, dim_out=128, num_heads=4),
            SAB(dim_in=128, dim_out=obs_act_dim, num_heads=1),
            # SAB(dim_in=64, dim_out=64, num_heads=4),
        )
        self.to(device)

    def forward(self, trajectory):
        # pos = self.position(x)
        # print(pos, x)
        x = self.dec(trajectory)
        # print(x.shape)
        return x


class TrajectoryEncoderPooling(nn.Module):
    def __init__(self, input_dim, device):
        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=input_dim, dim_out=128, num_heads=4),
            SAB(dim_in=128, dim_out=20, num_heads=4),
            # SAB(dim_in=64, dim_out=64, num_heads=4),
        )
        self.to(device)

    def forward(self, trajectory):
        # pos = self.position(x)
        # print(pos, x)
        x = self.enc(trajectory)
        # print(x.shape)
        return x


class TrajectoryDecoderPooling(nn.Module):
    def __init__(self, obs_act_dim, device):
        super().__init__()
        self.dec = nn.Sequential(
            SAB(dim_in=20, dim_out=128, num_heads=4),
            SAB(dim_in=128, dim_out=obs_act_dim, num_heads=1),
            # SAB(dim_in=64, dim_out=64, num_heads=4),
        )
        self.to(device)

    def forward(self, trajectory):
        # pos = self.position(x)
        # print(pos, x)
        x = self.dec(trajectory)
        # print(x.shape)
        return x


class EmbeddingPooling(nn.Module):
    def __init__(self, out_dim, device):
        super().__init__()
        self.dec = nn.Sequential(
                    PMA(dim=128, num_heads=4, num_seeds=1),
                    nn.Linear(in_features=128, out_features=out_dim)
                )
        self.to(device)

    def forward(self, features):
        output = self.dec(features)

        return output.squeeze()

class ProjectionHead(nn.Module):

    def __init__(self, input_dim, output_dim, device):
        super(ProjectionHead, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        self.to(device)

    def forward(self, feature):
        return self.projector(feature)
