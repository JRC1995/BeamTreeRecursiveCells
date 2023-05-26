
import math

import torch
from torch import nn
from torch.nn import init
import torch as T
import torch.nn.functional as F

class BinaryTreeLSTMLayer(nn.Module):

    def __init__(self, config):
        super(BinaryTreeLSTMLayer, self).__init__()
        self.hidden_dim = config["hidden_size"]
        self.dropout = config["dropout"]
        self.comp_linear = nn.Linear(in_features=2 * self.hidden_dim,
                                     out_features=5 * self.hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.comp_linear.weight.data)
        init.constant_(self.comp_linear.bias.data, val=0)


    def forward(self, l=None, r=None):
        hl, cl = l
        hr, cr = r
        hlr_cat = torch.cat([hl, hr], dim=2)
        treelstm_vector = self.comp_linear(hlr_cat)
        i, fl, fr, u, o = treelstm_vector.chunk(chunks=5, dim=2)
        c = (cl * (fl + 1).sigmoid() + cr * (fr + 1).sigmoid()
             + u.tanh() * i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c


class GumbelTreeLSTM(nn.Module):

    def __init__(self, config):
        super(GumbelTreeLSTM, self).__init__()
        self.config = config
        self.word_dim = config["hidden_size"]
        self.hidden_dim = config["hidden_size"]

        self.word_linear = nn.Linear(in_features=self.word_dim,
                                     out_features=2 * self.hidden_dim)

        self.treelstm_layer = BinaryTreeLSTMLayer(config)
        #self.comp_query = nn.Parameter(torch.FloatTensor(self.hidden_dim))
        if self.config["conv_decision"]:
            self.decide_linear = nn.Linear(5 * self.hidden_dim, 1)
        else:
            self.decide_linear = nn.Linear(self.hidden_dim, 1)


    @staticmethod
    def update_state(old_state, new_state, done_mask):
        old_h, old_c = old_state
        new_h, new_c = new_state
        done_mask = done_mask.float().unsqueeze(1).unsqueeze(2)
        h = done_mask * new_h + (1 - done_mask) * old_h[:, :-1, :]
        c = done_mask * new_c + (1 - done_mask) * old_c[:, :-1, :]
        return h, c




    def masked_softmax(self, logits, mask=None, dim=-1):
        eps = 1e-20
        probs = F.softmax(logits, dim=dim)
        if mask is not None:
            mask = mask.float()
            probs = probs * mask + eps
            probs = probs / probs.sum(dim, keepdim=True)
        return probs


    def st_gumbel_softmax(self, logits, temperature=1.0, mask=None):
        eps = 1e-20
        N, S = logits.size()

        if self.training:
            u = logits.data.new(*logits.size()).uniform_()
            gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
        else:
            gumbel_noise = 0
        y = logits + gumbel_noise

        y_ = self.masked_softmax(logits=y / temperature, mask=mask)
        y_argmax = y_.max(dim=-1)[1]
        y_hard = F.one_hot(y_argmax, num_classes=y.size(-1)).float()

        y = y_

        assert y.size() == (N, S)

        y = (y_hard - y).detach() + y
        return y

    def select_composition(self, old_state, new_state, mask):
        new_h, new_c = new_state
        old_h, old_c = old_state
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]
        old_c_left, old_c_right = old_c[:, :-1, :], old_c[:, 1:, :]


        if self.config["conv_decision"]:
            N, S, D = new_h.size()

            PAD = T.zeros(N, 1, D).float().to(new_h.device)
            new_h = new_h * mask.unsqueeze(-1)

            new_h_left1 = T.cat([PAD, new_h[:, 0:-1, :]], dim=1)
            new_h_left2 = T.cat([PAD, new_h_left1[:, 0:-1, :]], dim=1)
            new_h_right1 = T.cat([new_h[:, 1:, :], PAD], dim=1)
            new_h_right2 = T.cat([new_h_right1[:, 1:, :], PAD], dim=1)

            windowed_seq = T.cat([new_h_left2, new_h_left1, new_h, new_h_right1, new_h_right2], dim=-1)
            assert windowed_seq.size() == (N, S, 5 * D)

            comp_weights = self.decide_linear(windowed_seq).squeeze(-1)
        else:
            comp_weights = self.decide_linear(new_h).squeeze(-1)

        """
        comp_weights = (self.comp_query * new_h).sum(-1)
        comp_weights = comp_weights / math.sqrt(self.hidden_dim)
        """

        select_mask = self.st_gumbel_softmax(logits=comp_weights,
                                             mask=mask)

        select_mask_expand = select_mask.unsqueeze(2).expand_as(new_h)
        select_mask_cumsum = select_mask.cumsum(1)
        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(2).expand_as(old_h_left)
        right_mask = select_mask_cumsum - select_mask
        right_mask_expand = right_mask.unsqueeze(2).expand_as(old_h_right)
        new_h = (select_mask_expand * new_h
                 + left_mask_expand * old_h_left
                 + right_mask_expand * old_h_right)
        new_c = (select_mask_expand * new_c
                 + left_mask_expand * old_c_left
                 + right_mask_expand * old_c_right)
        selected_h = (select_mask_expand * new_h).sum(1)
        return new_h, new_c, select_mask, selected_h

    def forward(self, input, input_mask):
        sequence0 = input
        max_depth = input.size(1)
        length_mask = input_mask
        select_masks = []

        state = self.word_linear(input)
        state = state.chunk(chunks=2, dim=2)

        for i in range(max_depth - 1):
            h, c = state
            l = (h[:, :-1, :], c[:, :-1, :])
            r = (h[:, 1:, :], c[:, 1:, :])
            new_state = self.treelstm_layer(l=l, r=r)
            if i < max_depth - 2:
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                new_h, new_c, select_mask, selected_h = self.select_composition(
                    old_state=state, new_state=new_state,
                    mask=length_mask[:, i + 1:])
                new_state = (new_h, new_c)
                select_masks.append(select_mask)

            done_mask = length_mask[:, i + 1]
            state = self.update_state(old_state=state, new_state=new_state,
                                      done_mask=done_mask)

        h, c = state

        assert h.size(1) == 1 and c.size(1) == 1

        sequence = sequence0
        global_state = h.squeeze(1)
        input_mask = input_mask.unsqueeze(-1)
        aux = None

        return {"sequence": sequence,
                "global_state": global_state,
                "input_mask": input_mask,
                "aux_loss": aux}


