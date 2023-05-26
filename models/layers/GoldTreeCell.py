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
        concated = torch.cat([l, r], dim=2)
        intermediate = F.gelu(self.wcell1(concated))
        intermediate = F.dropout(intermediate, p=self.dropout, training=self.training)
        contents = self.wcell2(intermediate)

        contents = contents.view(N, S, 4, D)
        gates = torch.sigmoid(contents[:, :, 0:3, :])
        parent = contents[:, :, 3, :]
        f1 = gates[..., 0, :]
        f2 = gates[..., 1, :]
        i = gates[..., 2, :]
        transition = self.LN2(f1 * l + f2 * r + i * parent)
        return transition


class GoldTreeCell(nn.Module):

    def __init__(self, config):
        super(GoldTreeCell, self).__init__()
        self.config = config
        self.word_dim = config["hidden_size"]
        self.hidden_dim = config["hidden_size"]

        self.word_linear = nn.Linear(in_features=self.word_dim,
                                     out_features=self.hidden_dim)

        self.treecell_layer = CellLayer(self.hidden_dim, 4 * self.hidden_dim, config["dropout"])
        self.LN = nn.LayerNorm(self.hidden_dim)

    @staticmethod
    def update_state(old_state, new_state, done_mask):
        old_h = old_state
        new_h = new_state
        done_mask = done_mask.float().unsqueeze(1).unsqueeze(2)
        h = done_mask * new_h + (1 - done_mask) * old_h[:, :-1, :]
        # c = done_mask * new_c + (1 - done_mask) * old_c[:, :-1, :]
        return h

    def masked_softmax(self, logits, mask=None, dim=-1):
        eps = 1e-20
        probs = F.softmax(logits, dim=dim)
        if mask is not None:
            mask = mask.float()
            probs = probs * mask + eps
            probs = probs / probs.sum(dim, keepdim=True)
        return probs


    def select_composition(self, old_state, new_state, mask, select_mask):
        new_h = new_state
        old_h = old_state
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]
        select_mask_expand = select_mask.unsqueeze(2).expand_as(new_h)
        select_mask_cumsum = select_mask.cumsum(1)
        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(2).expand_as(old_h_left)
        right_mask = select_mask_cumsum - select_mask
        right_mask_expand = right_mask.unsqueeze(2).expand_as(old_h_right)
        new_h = (select_mask_expand * new_h
                 + left_mask_expand * old_h_left
                 + right_mask_expand * old_h_right)
        selected_h = (select_mask_expand * new_h).sum(1)
        return new_h, select_mask, selected_h

    def forward(self, input, input_mask, parse_trees):
        max_depth = input.size(1)
        length_mask = input_mask
        select_masks = []

        state = self.LN(self.word_linear(input))
        for i in range(max_depth - 1):
            h = state
            l = h[:, :-1, :]
            r = h[:, 1:, :]
            new_state = self.treecell_layer(l=l, r=r)
            if i < max_depth - 2:
                S = l.size(1)
                select_mask = parse_trees[:, i, 0:S]
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                new_h, select_mask, selected_h = self.select_composition(
                    old_state=state, new_state=new_state,
                    mask=length_mask[:, i + 1:],
                    select_mask=select_mask)
                new_state = new_h
                select_masks.append(select_mask)
            done_mask = length_mask[:, i + 1]
            state = self.update_state(old_state=state, new_state=new_state,
                                      done_mask=done_mask)
        h = state

        sequence = input
        input_mask = input_mask.unsqueeze(-1)
        aux_loss = None
        global_state = h.squeeze(1)

        return {"sequence": sequence, "global_state": global_state, "input_mask": input_mask, "aux_loss": aux_loss}
