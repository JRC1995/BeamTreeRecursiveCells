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
        N, D = l.size()
        concated = torch.cat([l, r], dim=-1)
        intermediate = F.gelu(self.wcell1(concated))
        intermediate = F.dropout(intermediate, p=self.dropout, training=self.training)
        contents = self.wcell2(intermediate)

        contents = contents.view(N, 4, D)
        gates = torch.sigmoid(contents[..., 0:3, :])
        parent = contents[..., 3, :]
        f1 = gates[..., 0, :]
        f2 = gates[..., 1, :]
        i = gates[..., 2, :]
        transition = self.LN2(f1 * l + f2 * r + i * parent)
        assert transition.size() == (N, D)
        return transition


class BiCell(nn.Module):

    def __init__(self, config):
        super(BiCell, self).__init__()
        self.config = config
        self.word_dim = config["hidden_size"]
        self.hidden_dim = config["hidden_size"]
        self.bidirectional = config["bidirectional"]

        self.word_linear = nn.Linear(in_features=self.word_dim,
                                     out_features=self.hidden_dim)

        self.fcell = CellLayer(self.hidden_dim, 4 * self.hidden_dim, config["dropout"])
        if self.bidirectional:
            self.bcell = CellLayer(self.hidden_dim, 4 * self.hidden_dim, config["dropout"])

        self.LN = nn.LayerNorm(self.hidden_dim)
        self.initial_fh = nn.Parameter(T.zeros(self.hidden_dim).float())
        if self.bidirectional:
            self.initial_bh = nn.Parameter(T.zeros(self.hidden_dim).float())
            self.compress_linear1 = nn.Linear(in_features=2 * self.hidden_dim,
                                              out_features=self.hidden_dim)
            self.compress_linear2 = nn.Linear(in_features=2 * self.hidden_dim,
                                              out_features=self.hidden_dim)

    def forward(self, input, input_mask):
        input_mask = input_mask.unsqueeze(-1)
        state = self.LN(self.word_linear(input))
        N, S, D = state.size()
        assert input_mask.size() == (N, S, 1)
        h = self.initial_fh.view(1, D).repeat(N, 1)
        f_hs = []
        for i in range(S):
            m = input_mask[:, i, :]
            inp = state[:, i, :]
            assert h.size() == (N, D)
            assert inp.size() == (N, D)
            assert m.size() == (N, 1)
            h_ = self.fcell(l=h, r=inp)
            h = m * h_ + (1 - m) * h
            f_hs.append(h)

        fsequence = T.stack(f_hs, dim=1)
        sequence = fsequence
        global_state = h
        assert global_state.size() == (N, D)

        if self.bidirectional:
            b_hs = []
            h = self.initial_bh.view(1, D).repeat(N, 1)
            for i in range(S):
                m = input_mask[:, S - 1 - i, :]
                inp = state[:, S - 1 - i, :]
                assert h.size() == (N, D)
                assert inp.size() == (N, D)
                assert m.size() == (N, 1)
                h_ = self.bcell(l=h, r=inp)
                h = m * h_ + (1 - m) * h
                b_hs.append(h)

            global_state = T.cat([f_hs[-1], b_hs[-1]], dim=-1)
            assert global_state.size() == (N, 2 * D)

            global_state = self.compress_linear1(global_state)
            assert global_state.size() == (N, D)

            b_hs.reverse()
            bsequence = T.stack(b_hs, dim=1)
            sequence = T.cat([fsequence, bsequence], dim=-1)
            assert sequence.size() == (N, S, 2 * D)
            sequence = self.compress_linear2(sequence)

        assert sequence.size() == (N, S, D)
        aux_loss = None

        return {"sequence": sequence, "global_state": global_state, "input_mask": input_mask, "aux_loss": aux_loss}
