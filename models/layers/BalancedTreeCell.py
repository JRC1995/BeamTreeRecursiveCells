import math
import torch
from torch import nn
from torch.nn import init
import torch as T
import torch.nn.functional as F


class CellLayer(nn.Module):

    def __init__(self, hidden_dim, cell_hidden_dim, dropout):
        super(CellLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.wcell1 = nn.Linear(2 * hidden_dim, cell_hidden_dim)
        self.wcell2 = nn.Linear(cell_hidden_dim, 4 * hidden_dim)
        self.LN2 = nn.LayerNorm(hidden_dim)
        self.dropout = dropout


    def forward(self, l=None, r=None):
        N, S, D = l.size()
        concated = torch.cat([l, r], dim=-1)
        intermediate = F.gelu(self.wcell1(concated))
        intermediate = F.dropout(intermediate, p=self.dropout, training=self.training)
        contents = self.wcell2(intermediate)

        contents = contents.view(N, S, 4, D)
        gates = torch.sigmoid(contents[..., 0:3, :])
        parent = contents[..., 3, :]
        f1 = gates[..., 0, :]
        f2 = gates[..., 1, :]
        i = gates[..., 2, :]
        transition = self.LN2(f1 * l + f2 * r + i * parent)
        assert transition.size() == (N, S, D)
        return transition


class BalancedTreeCell(nn.Module):

    def __init__(self, config):
        super(BalancedTreeCell, self).__init__()
        self.config = config
        self.word_dim = config["hidden_size"]
        self.hidden_dim = config["hidden_size"]

        self.word_linear = nn.Linear(in_features=self.word_dim,
                                     out_features=self.hidden_dim)

        self.tree_cell = CellLayer(self.hidden_dim, 4 * self.hidden_dim, config["dropout"])
        self.LN = nn.LayerNorm(self.hidden_dim)


    def forward(self, input, input_mask):
        input_mask = input_mask.unsqueeze(-1)
        state = self.LN(self.word_linear(input))
        N, S, D = state.size()

        assert input_mask.size() == (N, S, 1)

        for i in range(S):
            S_ = state.size(1)
            if S_ % 2 != 0:
                PAD = T.zeros(N, 1, D).float().to(state.device)
                state = T.cat([state, PAD], dim=1)
                input_mask = T.cat([input_mask, PAD[..., 0].unsqueeze(-1)], dim=1)
                S_ = S_ + 1
                assert S_ % 2 == 0
                assert state.size() == (N, S_, D)
                assert input_mask.size() == (N, S_, 1)

            if S_ == 1:
                break
            else:
                l = []
                r = []
                ml = []
                mr = []
                for j in range(S_):
                    if j % 2 == 0:
                        l.append(state[:, j, :])
                        ml.append(input_mask[:, j, :])
                    else:
                        r.append(state[:, j, :])
                        mr.append(input_mask[:, j, :])

                l = T.stack(l, dim=1)
                r = T.stack(r, dim=1)
                mr = T.stack(mr, dim=1)
                ml = T.stack(ml, dim=1)

                assert l.size() == (N, S_ // 2, D)
                assert r.size() == (N, S_ // 2, D)
                assert ml.size() == (N, S_ // 2, 1)
                assert mr.size() == (N, S_ // 2, 1)

                state_ = self.tree_cell(l, r)
                state = mr * state_ + (1 - mr) * l
                input_mask = ml

        assert state.size() == (N, 1, D)
        global_state = state.squeeze(1)
        assert global_state.size() == (N, D)
        aux_loss = None

        return {"sequence": input,
                "global_state": global_state,
                "input_mask": input_mask,
                "aux_loss": aux_loss}
