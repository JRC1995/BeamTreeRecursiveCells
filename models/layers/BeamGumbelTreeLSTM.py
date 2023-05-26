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
        N, B, S, D = l.size()
        concated = torch.cat([l, r], dim=-1)
        intermediate = F.gelu(self.wcell1(concated))
        intermediate = F.dropout(intermediate, p=self.dropout, training=self.training)
        contents = self.wcell2(intermediate)

        contents = contents.view(N, B, S, 4, D)
        gates = torch.sigmoid(contents[..., 0:3, :])
        parent = contents[..., 3, :]
        f1 = gates[..., 0, :]
        f2 = gates[..., 1, :]
        i = gates[..., 2, :]
        transition = self.LN2(f1 * l + f2 * r + i * parent)
        return transition


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
        hl = l[..., 0:self.hidden_dim]
        cl = l[..., self.hidden_dim:]
        hr = r[..., 0:self.hidden_dim]
        cr = r[..., self.hidden_dim:]
        hlr_cat = torch.cat([hl, hr], dim=-1)
        treelstm_vector = self.comp_linear(hlr_cat)
        i, fl, fr, u, o = treelstm_vector.chunk(chunks=5, dim=-1)
        c = (cl * (fl + 1).sigmoid() + cr * (fr + 1).sigmoid()
             + u.tanh() * i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return T.cat([h, c], dim=-1)


class BeamGumbelTreeLSTM(nn.Module):
    def __init__(self, config):
        super(BeamGumbelTreeLSTM, self).__init__()
        self.config = config
        self.word_dim = config["hidden_size"]
        self.hidden_dim = config["hidden_size"]
        self.beam_size = config["beam_size"]
        self.diffop1 = config["diffop1"]
        self.diffop2 = config["diffop2"]

        self.word_linear = nn.Linear(in_features=self.word_dim,
                                     out_features=2 * self.hidden_dim)

        self.treecell_layer = BinaryTreeLSTMLayer(config)
        if self.config["conv_decision"]:
            self.decide_linear = nn.Linear(5 * self.hidden_dim, 1)
        else:
            self.decide_linear = nn.Linear(self.hidden_dim, 1)
        # self.comp_query = nn.Parameter(torch.FloatTensor(self.hidden_dim))

    @staticmethod
    def update_state(old_state, new_state, done_mask):
        old_h = old_state
        new_h = new_state
        N, B, S, D = new_h.size()
        assert old_h.size() == (N, B, S + 1, D)
        done_mask = done_mask.view(N, 1, 1, 1)

        h = done_mask * new_h + (1 - done_mask) * old_h[..., :-1, :]
        return h

    def masked_softmax(self, logits, mask=None, dim=-1):
        eps = 1e-20
        probs = F.softmax(logits, dim=dim)
        if mask is not None:
            mask = mask.float()
            probs = probs * mask + eps
            probs = probs / probs.sum(dim, keepdim=True)
        return probs


    def st_gumbel_softmax(self, logits, select_k=1, temperature=1.0, mask=None, training=True):
        eps = 1e-20
        N, S = logits.size()

        if (self.config["stochastic"] and (self.training or self.config["test_time_stochastic"])) or training:
            u = logits.data.new(*logits.size()).uniform_()
            gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
        else:
            gumbel_noise = 0

        y = logits + gumbel_noise

        y_ = self.masked_softmax(logits=y / temperature, mask=mask)
        y_argmax = T.topk(y_, dim=-1, k=select_k)[1]
        assert y_argmax.size() == (N, select_k)

        y_hard = F.one_hot(y_argmax, num_classes=S).float()
        assert y_hard.size() == (N, select_k, S)

        if not training:
            return y_hard, y_.unsqueeze(1).repeat(1, select_k, 1)
        else:
            y = y_.unsqueeze(1).repeat(1, select_k, 1)
            assert y.size() == (N, select_k, S)

            assert y.size() == (N, select_k, S)
            assert y_hard.size() == (N, select_k, S)

            y_hard = (y_hard - y).detach() + y
            return y_hard, y

    def select_composition(self, old_state, new_state, mask, accu_scores, beam_mask):
        new_h = new_state
        old_h = old_state

        N, B, S, D = new_h.size()
        assert accu_scores.size() == (N, B)
        assert mask.size() == (N, S)
        assert beam_mask.size() == (N, B)

        if self.config["conv_decision"]:
            raise ValueError("Not properly implemented")
            PAD = T.zeros(N, B, 1, D).float().to(new_h.device)
            new_h = new_h * mask.unsqueeze(1).unsqueeze(-1)

            new_h_left1 = T.cat([PAD, new_h[..., 0:-1, :]], dim=-2)
            new_h_left2 = T.cat([PAD, new_h_left1[..., 0:-1, :]], dim=-2)
            new_h_right1 = T.cat([new_h[..., 1:, :], PAD], dim=-2)
            new_h_right2 = T.cat([new_h_right1[..., 1:, :], PAD], dim=-2)

            windowed_seq = T.cat([new_h_left2, new_h_left1, new_h, new_h_right1, new_h_right2], dim=-1)
            assert windowed_seq.size() == (N, B, S, 5 * D)

            comp_weights = self.decide_linear(windowed_seq).squeeze(-1)
        else:
            comp_weights = self.decide_linear(new_h[..., 0:D // 2]).squeeze(-1)  # / math.sqrt(self.hidden_dim)

        topk = min(S, self.beam_size)

        training = self.training if self.diffop1 else False

        select_mask, soft_scores = self.st_gumbel_softmax(logits=comp_weights.view(N * B, S),
                                                          temperature=1,
                                                          mask=mask.view(N, 1, S).repeat(1, B, 1).view(N * B, S),
                                                          select_k=topk,
                                                          training=training)

        soft_scores = F.softmax(comp_weights, dim=-1)
        assert soft_scores.size() == (N, B, S)
        soft_scores = mask.unsqueeze(1) * soft_scores + 1e-20
        soft_scores = soft_scores / T.sum(soft_scores, dim=-1, keepdim=True)
        soft_scores = soft_scores.unsqueeze(2)

        assert select_mask.size() == (N * B, topk, S)
        select_mask = select_mask.view(N, B, topk, S)
        assert soft_scores.size() == (N, B, 1, S)
        # soft_scores = soft_scores.view(N, B, topk, S)
        new_scores = T.log(T.sum(select_mask * soft_scores, dim=-1) + 1e-20)
        assert new_scores.size() == (N, B, topk)

        done_mask = 1 - mask[:, 0].view(N, 1, 1).repeat(1, B, 1)
        if topk == 1:
            done_topk = T.ones(N, B, topk).float().to(mask.device)
        else:
            done_topk = T.cat([T.ones(N, B, 1).float().to(mask.device),
                               T.zeros(N, B, topk - 1).float().to(mask.device)], dim=-1)
        assert done_topk.size() == (N, B, topk)

        not_done_topk = T.ones(N, B, topk).float().to(mask.device)
        new_beam_mask = done_mask * done_topk + (1 - done_mask) * not_done_topk
        beam_mask = beam_mask.unsqueeze(-1) * new_beam_mask

        assert beam_mask.size() == (N, B, topk)
        beam_mask = beam_mask.view(N, B * topk)

        accu_scores = accu_scores.view(N, B, 1) + new_scores
        accu_scores = accu_scores.view(N, B * topk)
        # accu_scores = T.clip(accu_scores, min=-999999)

        select_mask = select_mask.view(N, B * topk, S)

        new_h = new_h.unsqueeze(2).repeat(1, 1, topk, 1, 1)
        old_h = old_h.unsqueeze(2).repeat(1, 1, topk, 1, 1)
        assert new_h.size() == (N, B, topk, S, D)
        assert old_h.size() == (N, B, topk, S + 1, D)

        new_h = new_h.view(N, B * topk, S, D)
        old_h = old_h.view(N, B * topk, S + 1, D)

        if (B * topk) > self.beam_size:
            B2 = self.beam_size
            assert accu_scores.size() == beam_mask.size()
            training = self.training if self.diffop2 else False
            beam_select_mask, _ = self.st_gumbel_softmax(logits=accu_scores,
                                                         temperature=1,
                                                         mask=beam_mask,
                                                         select_k=B2,
                                                         training=training)
            assert beam_select_mask.size() == (N, B2, B * topk)

            new_h = T.matmul(beam_select_mask, new_h.view(N, B * topk, -1))
            new_h = new_h.view(N, B2, S, D)

            old_h = T.matmul(beam_select_mask, old_h.view(N, B * topk, -1))
            old_h = old_h.view(N, B2, S + 1, D)

            select_mask = T.matmul(beam_select_mask, select_mask)
            assert select_mask.size() == (N, B2, S)

            accu_scores = T.matmul(beam_select_mask, accu_scores.unsqueeze(-1)).squeeze(-1)
            assert accu_scores.size() == (N, B2)

            beam_mask = T.matmul(beam_select_mask, beam_mask.unsqueeze(-1)).squeeze(-1)
            assert beam_mask.size() == (N, B2)
        else:
            B2 = B * topk

        select_mask_expand = select_mask.unsqueeze(-1)
        select_mask_cumsum = select_mask.cumsum(-1)

        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(-1)

        right_mask = select_mask_cumsum - select_mask
        right_mask_expand = right_mask.unsqueeze(-1)

        old_h_left, old_h_right = old_h[..., :-1, :], old_h[..., 1:, :]

        assert select_mask_expand.size() == (N, B2, S, 1)
        assert left_mask_expand.size() == (N, B2, S, 1)
        assert right_mask_expand.size() == (N, B2, S, 1)
        assert new_h.size() == (N, B2, S, D)
        assert old_h_left.size() == (N, B2, S, D)
        assert old_h_right.size() == (N, B2, S, D)

        new_h = (select_mask_expand * new_h
                 + left_mask_expand * old_h_left
                 + right_mask_expand * old_h_right)

        return new_h, old_h, accu_scores, beam_mask

    def forward(self, input, input_mask):
        max_depth = input.size(1)
        length_mask = input_mask

        state = self.word_linear(input)
        N, S, D = state.size()
        B = 1
        state = state.unsqueeze(1)
        assert state.size() == (N, B, S, D)

        accu_scores = T.zeros(N, B).float().to(state.device)
        beam_mask = T.ones(N, B).float().to(state.device)

        for i in range(max_depth - 1):
            S = state.size(-2)
            B = state.size(1)
            h = state
            assert h.size() == (N, B, S, D)
            l = h[:, :, :-1, :]
            r = h[:, :, 1:, :]
            assert l.size() == (N, B, S - 1, D)
            assert r.size() == (N, B, S - 1, D)
            new_state = self.treecell_layer(l=l, r=r)
            if i < max_depth - 2:
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                new_h, old_h, accu_scores, beam_mask = self.select_composition(old_state=state, new_state=new_state,
                                                                               mask=length_mask[:, i + 1:],
                                                                               accu_scores=accu_scores,
                                                                               beam_mask=beam_mask)
                new_state = new_h
                state = old_h
            done_mask = length_mask[:, i + 1]
            state = self.update_state(old_state=state, new_state=new_state,
                                      done_mask=done_mask)
        h = state[..., 0:self.hidden_dim]
        sequence = input
        input_mask = input_mask.unsqueeze(-1)
        aux_loss = None

        N, B, S, D = h.size()
        assert S == 1
        h = h.squeeze(-2)
        assert h.size() == (N, B, D)
        assert accu_scores.size() == (N, B)
        assert beam_mask.size() == (N, B)
        normed_scores = F.softmax(beam_mask * accu_scores + (1 - beam_mask) * -999999, dim=-1)

        global_state = T.sum(normed_scores.unsqueeze(-1) * h, dim=1)
        assert global_state.size() == (N, D)

        return {"sequence": sequence, "global_state": global_state, "input_mask": input_mask, "aux_loss": aux_loss}
