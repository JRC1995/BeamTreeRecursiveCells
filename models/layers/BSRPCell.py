import torch as T
import torch.nn as nn
import torch.nn.functional as F
from diffsort import DiffSortNet
from models.activations import entmax15


class BSRPCell(nn.Module):
    def __init__(self, config):
        super(BSRPCell, self).__init__()

        self.hidden_size = config["hidden_size"]
        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.cell_hidden_size = 4 * config["hidden_size"]
        self.hidden_dropout = config["dropout"]
        self.config = config
        self.tree_num = config["beam_size"]
        self.keep_lost_info = False
        # self.sorter = DiffSortNet('bitonic', self.max_stack_num, steepness=10, device=self.config["device"])

        self.initial_transform_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.LN1 = nn.LayerNorm(self.hidden_size)

        self.scorer = nn.Linear(self.hidden_size, 1)
        self.conv_layer = nn.Linear(3 * self.hidden_size, self.hidden_size)
        self.START = nn.Parameter(T.randn(self.hidden_size))

        self.wcell1 = nn.Linear(2 * self.hidden_size, self.cell_hidden_size)
        self.wcell2 = nn.Linear(self.cell_hidden_size, 4 * self.hidden_size)
        self.LN2 = nn.LayerNorm(self.hidden_size)
        self.eps = 1e-8

    # %%
    def initial_transform(self, sequence):
        sequence = self.LN1(self.initial_transform_layer(sequence))
        return sequence

    def compose(self, child1, child2):
        N, S, D = child1.size()

        concated = T.cat([child1, child2], dim=-1)
        assert concated.size() == (N, S, 2 * D)

        intermediate = F.gelu(self.wcell1(concated))
        intermediate = F.dropout(intermediate, p=self.hidden_dropout, training=self.training)
        contents = self.wcell2(intermediate)
        contents = contents.view(N, S, 4, D)
        gates = T.sigmoid(contents[..., 0:3, :])
        parent = contents[..., 3, :]
        f1 = gates[..., 0, :]
        f2 = gates[..., 1, :]
        i = gates[..., 2, :]

        transition = self.LN2(f1 * child1 + f2 * child2 + i * parent)

        return transition

    # %%
    def sum_normalize(self, logits, dim=-1):
        return logits / T.sum(logits + self.eps, keepdim=True, dim=dim)

    # %%
    def encoder_block(self, sequence, input_mask):
        """
        N = Batch Size
        S = Sequence Size
        """
        N, S, D = sequence.size()
        input_mask = input_mask.view(N, S, 1)
        sequence = self.initial_transform(sequence)

        if S == 1:
            global_state = sequence[:, 0, :]
            assert global_state.size() == (N, D)
        elif S == 2:
            item1 = sequence[:, 0, :].unsqueeze(1)
            item2 = sequence[:, 1, :].unsqueeze(1)
            reduce_items = self.compose(item1, item2)
            input_mask_ = input_mask[:, 1, :].unsqueeze(1)
            reduce_items = input_mask_ * reduce_items + (1-input_mask_) * item1
            global_state = reduce_items.squeeze(1)
            assert global_state.size() == (N, D)
        else:
            stack_size = 2
            stack_num = 1
            stacks = T.stack([sequence[:, 0, :], sequence[:, 1, :]], dim=1).unsqueeze(1)
            assert stacks.size() == (N, stack_num, stack_size, D)

            stacks_last_mask = T.stack([T.zeros(N, stack_num, 1).float().to(sequence.device),
                                        T.ones(N, stack_num, 1).float().to(sequence.device)], dim=-2)

            assert stacks_last_mask.size() == (N, stack_num, stack_size, 1)
            stacks_laster_mask = T.stack([T.ones(N, stack_num, 1).float().to(sequence.device),
                                          T.zeros(N, stack_num, 1).float().to(sequence.device)], dim=-2)
            assert stacks_laster_mask.size() == (N, stack_num, stack_size, 1)
            im = input_mask[:, 1, :].unsqueeze(1).unsqueeze(2)
            assert im.size() == (N, 1, 1, 1)

            stacks_last_mask = im * stacks_last_mask + (1 - im) * stacks_laster_mask
            stack_scores = T.zeros(N, stack_num).float().to(sequence.device)
            curr_items = sequence[:, 2:, :].unsqueeze(1).repeat(1, stack_num, 1, 1)
            curr_mask = input_mask[:, 2:, :].unsqueeze(1).repeat(1, stack_num, 1, 1)
            assert curr_items.size() == (N, stack_num, S - 2, D)
            assert curr_mask.size() == (N, stack_num, S - 2, 1)

            START = self.START.view(1, 1, D)


            for t in range(2, 2 * S):
                stack_num = stacks.size(1)
                stack_size = stacks.size(2)
                curr_item = curr_items[:, :, 0, :]
                cm = curr_mask[:, :, 0, :]
                assert curr_item.size() == (N, stack_num, D)
                assert cm.size() == (N, stack_num, 1)

                last_items = T.sum(stacks_last_mask * stacks, dim=-2)
                assert last_items.size() == (N, stack_num, D)

                stacks_laster_mask = T.cat([stacks_last_mask[:, :, 1:, :],
                                            T.zeros(N, stack_num, 1, 1).float().to(sequence.device)], dim=-2)
                assert stacks_laster_mask.size() == (N, stack_num, stack_size, 1)
                laster_prob = T.sum(stacks_laster_mask, dim=-2)
                assert laster_prob.size() == (N, stack_num, 1)

                laster_items = T.sum(stacks_laster_mask * stacks, dim=-2)
                assert laster_items.size() == (N, stack_num, D)

                laster_items = laster_prob * laster_items + (1-laster_prob) * START

                windowed_sequence = T.cat([laster_items, last_items, curr_item], dim=-1)
                assert windowed_sequence.size() == (N, stack_num, 3 * D)
                decision_scores = T.sigmoid(self.scorer(F.gelu(self.conv_layer(windowed_sequence))))
                assert decision_scores.size() == (N, stack_num, 1)

                reduce_scores = decision_scores[..., 0]
                shift_scores = 1 - decision_scores[..., 0]

                both_zeros = T.where((laster_prob.squeeze(-1) == 0) & (cm.squeeze(-1) == 0),
                                     T.ones(N, stack_num).float().to(decision_scores.device),
                                     T.zeros(N, stack_num).float().to(decision_scores.device))

                if T.sum(1-both_zeros) == 0.0:
                    break

                reduce_scores = cm.squeeze(-1) * reduce_scores + (1-cm.squeeze(-1))
                reduce_scores = laster_prob.squeeze(-1) * reduce_scores
                reduce_scores = both_zeros + (1-both_zeros) * reduce_scores
                reduce_scores = T.log(reduce_scores + self.eps)

                shift_scores = laster_prob.squeeze(-1) * shift_scores + (1-laster_prob.squeeze(-1))
                shift_scores = cm.squeeze(-1) * shift_scores
                shift_scores = (1-both_zeros) * shift_scores
                shift_scores = T.log(shift_scores + self.eps)


                reduce_items = self.compose(laster_items, last_items)
                assert reduce_items.size() == (N, stack_num, D)

                reduce_stacks = stacks_laster_mask * reduce_items.unsqueeze(-2) + (1 - stacks_laster_mask) * stacks
                reduce_stacks = T.cat([reduce_stacks,
                                       T.zeros(N, stack_num, 1, D).float().to(sequence.device)], dim=-2)
                assert reduce_stacks.size() == (N, stack_num, stack_size + 1, D)

                reduce_stacks_last_mask = laster_prob.unsqueeze(-1) * stacks_laster_mask \
                                          + (1 - laster_prob.unsqueeze(-1)) * stacks_last_mask

                reduce_stacks_last_mask = T.cat([reduce_stacks_last_mask,
                                                 T.zeros(N, stack_num, 1, 1).float().to(sequence.device)], dim=-2)
                assert reduce_stacks_last_mask.size() == (N, stack_num, stack_size + 1, 1)

                reduce_curr_items = curr_items.clone()
                reduce_curr_mask = curr_mask.clone()
                reduce_stack_scores = stack_scores + reduce_scores
                assert reduce_stack_scores.size() == (N, stack_num)

                shift_stacks_initial = T.cat([stacks,
                                              T.zeros(N, stack_num, 1, D).float().to(sequence.device)], dim=-2)
                assert shift_stacks_initial.size() == (N, stack_num, stack_size + 1, D)

                shift_stacks_last_mask_initial = T.cat([stacks_last_mask,
                                                        T.zeros(N, stack_num, 1, 1).float().to(sequence.device)],
                                                       dim=-2)
                assert shift_stacks_last_mask_initial.size() == (N, stack_num, stack_size + 1, 1)

                shift_stacks_last_mask = T.cat([T.zeros(N, stack_num, 1, 1).float().to(sequence.device),
                                                stacks_last_mask], dim=-2)
                assert shift_stacks_last_mask.size() == (N, stack_num, stack_size + 1, 1)
                shift_stacks_last_mask = cm.unsqueeze(-1) * shift_stacks_last_mask \
                                         + (1 - cm.unsqueeze(-1)) * shift_stacks_last_mask_initial
                shift_stacks = shift_stacks_last_mask * curr_item.unsqueeze(-2) \
                               + (1 - shift_stacks_last_mask) * shift_stacks_initial
                shift_stacks = cm.unsqueeze(-1) * shift_stacks + (1 - cm.unsqueeze(-1)) * shift_stacks_initial

                shift_curr_items = T.cat([curr_items, T.zeros(N, stack_num, 1, D).float().to(sequence.device)], dim=-2)
                shift_curr_items = shift_curr_items[:, :, 1:, :]
                shift_curr_mask = T.cat([curr_mask, T.zeros(N, stack_num, 1, 1).float().to(sequence.device)], dim=-2)
                shift_curr_mask = shift_curr_mask[:, :, 1:, :]

                shift_stack_scores = stack_scores + shift_scores
                assert shift_stack_scores.size() == (N, stack_num)

                stacks = T.cat([reduce_stacks, shift_stacks], dim=1)
                assert stacks.size() == (N, 2 * stack_num, stack_size + 1, D)
                stacks = stacks[:, :, 0:S, :]

                stacks_last_mask = T.cat([reduce_stacks_last_mask, shift_stacks_last_mask], dim=1)
                assert stacks_last_mask.size() == (N, 2 * stack_num, stack_size + 1, 1)
                stacks_last_mask = stacks_last_mask[:, :, 0:S, :]

                curr_items = T.cat([reduce_curr_items, shift_curr_items], dim=1)
                assert curr_items.size() == (N, 2 * stack_num, S - 2, D)

                curr_mask = T.cat([reduce_curr_mask, shift_curr_mask], dim=1)
                assert curr_mask.size() == (N, 2 * stack_num, S - 2, 1)

                stack_scores = T.cat([reduce_scores, shift_scores], dim=1)
                assert stack_scores.size() == (N, 2 * stack_num)

                stack_size = min(S, stack_size + 1)

                if (2 * stack_num) >= self.tree_num:

                    sort_idx = T.argsort(stack_scores, dim=- 1, descending=False)
                    assert sort_idx.size() == (N, 2 * stack_num)
                    permute_matrix = F.one_hot(sort_idx, num_classes=2 * stack_num).float()
                    assert permute_matrix.size() == (N, 2 * stack_num, 2 * stack_num)

                    rest_stack_num = 2 * stack_num - self.tree_num + 1

                    stacks = T.matmul(permute_matrix, stacks.view(N, 2 * stack_num, -1))
                    stacks = stacks.view(N, 2 * stack_num, stack_size, D)
                    if self.keep_lost_info:
                        rest_stacks = stacks[:, 0:-self.tree_num + 1, ...]
                        assert rest_stacks.size() == (N, rest_stack_num, stack_size, D)
                        stacks = stacks[:, -self.tree_num + 1:, ...]
                        assert stacks.size() == (N, self.tree_num - 1, stack_size, D)
                    else:
                        stacks = stacks[:, -self.tree_num:, ...]
                        assert stacks.size() == (N, self.tree_num, stack_size, D)

                    stacks_last_mask = T.matmul(permute_matrix, stacks_last_mask.view(N, 2 * stack_num, -1))
                    stacks_last_mask = stacks_last_mask.view(N, 2 * stack_num, stack_size, 1)
                    if self.keep_lost_info:
                        rest_stacks_last_mask = stacks_last_mask[:, 0:self.tree_num + 1, ...]
                        assert rest_stacks_last_mask.size() == (N, rest_stack_num, stack_size, 1)
                        stacks_last_mask = stacks_last_mask[:, -self.tree_num + 1:, ...]
                        assert stacks_last_mask.size() == (N, self.tree_num - 1, stack_size, 1)
                    else:
                        stacks_last_mask = stacks_last_mask[:, -self.tree_num:, ...]
                        assert stacks_last_mask.size() == (N, self.tree_num, stack_size, 1)

                    curr_items = T.matmul(permute_matrix, curr_items.view(N, 2 * stack_num, -1))
                    curr_items = curr_items.view(N, 2 * stack_num, S - 2, D)
                    if self.keep_lost_info:
                        rest_curr_items = curr_items[:, 0:self.tree_num + 1, ...]
                        assert rest_curr_items.size() == (N, rest_stack_num, S - 2, D)
                        curr_items = curr_items[:, -self.tree_num + 1:, ...]
                        assert curr_items.size() == (N, self.tree_num - 1, S - 2, D)
                    else:
                        curr_items = curr_items[:, -self.tree_num:, ...]
                        assert curr_items.size() == (N, self.tree_num, S - 2, D)

                    curr_mask = T.matmul(permute_matrix, curr_mask.view(N, 2 * stack_num, -1))
                    curr_mask = curr_mask.view(N, 2 * stack_num, S - 2, 1)
                    if self.keep_lost_info:
                        rest_curr_mask = curr_mask[:, 0:self.tree_num + 1, ...]
                        assert rest_curr_mask.size() == (N, rest_stack_num, S - 2, 1)
                        curr_mask = curr_mask[:, -self.tree_num + 1:, ...]
                        assert curr_mask.size() == (N, self.tree_num - 1, S - 2, 1)
                    else:
                        curr_mask = curr_mask[:, -self.tree_num:, ...]
                        assert curr_mask.size() == (N, self.tree_num, S - 2, 1)

                    stack_socres = T.matmul(permute_matrix, stack_scores.view(N, 2 * stack_num, -1))
                    stack_scores = stack_socres.view(N, 2 * stack_num)
                    if self.keep_lost_info:
                        rest_stack_scores = stack_scores[:, 0:-self.tree_num + 1]
                        assert rest_stack_scores.size() == (N, rest_stack_num)
                        stack_scores = stack_scores[:, -self.tree_num + 1:]
                        assert stack_scores.size() == (N, self.tree_num - 1)
                    else:
                        stack_scores = stack_scores[:, -self.tree_num:]
                        assert stack_scores.size() == (N, self.tree_num)

                    if self.keep_lost_info:
                        normed_rest_stack_scores = F.softmax(rest_stack_scores, dim=-1)
                        rest_stacks = T.sum(normed_rest_stack_scores.view(N, rest_stack_num, 1, 1) \
                                            * rest_stacks, dim=1).unsqueeze(1)
                        assert rest_stacks.size() == (N, 1, stack_size, D)
                        stacks = T.cat([stacks, rest_stacks], dim=1)
                        assert stacks.size() == (N, self.tree_num, stack_size, D)

                        rest_stacks_last_mask = T.sum(normed_rest_stack_scores.view(N, rest_stack_num, 1, 1) \
                                                      * rest_stacks_last_mask, dim=1).unsqueeze(1)
                        assert rest_stacks_last_mask.size() == (N, 1, stack_size, 1)
                        stacks_last_mask = T.cat([stacks_last_mask, rest_stacks_last_mask], dim=1)
                        assert stacks_last_mask.size() == (N, self.tree_num, stack_size, 1)

                        rest_stack_scores = T.sum(normed_rest_stack_scores * rest_stack_scores, dim=1).unsqueeze(1)
                        assert rest_stack_scores.size() == (N, 1)
                        stack_scores = T.cat([stack_scores, rest_stack_scores], dim=1)
                        assert stack_scores.size() == (N, self.tree_num)

                        rest_curr_items = T.sum(normed_rest_stack_scores.view(N, rest_stack_num, 1, 1) \
                                                * rest_curr_items, dim=1).unsqueeze(1)

                        curr_items = T.cat([curr_items, rest_curr_items], dim=1)
                        assert curr_items.size() == (N, self.tree_num, S - 2, D)

                        rest_curr_mask = T.sum(normed_rest_stack_scores.view(N, rest_stack_num, 1, 1) \
                                               * rest_curr_mask, dim=1).unsqueeze(1)

                        curr_mask = T.cat([curr_mask, rest_curr_mask], dim=1)
                        assert curr_mask.size() == (N, self.tree_num, S - 2, 1)



            stack_num = stacks.size(1)
            stack_size = stacks.size(2)
            assert stacks.size() == (N, stack_num, stack_size, D)
            h = T.sum(stacks * stacks_last_mask, dim=-2)
            assert h.size() == (N, stack_num, D)
            stack_scores = F.softmax(stack_scores, dim=1)
            assert stack_scores.size() == (N, stack_num)
            global_state = T.sum(stack_scores.unsqueeze(-1) * h, dim=1)
            assert global_state.size() == (N, D)

        aux_loss = None

        return sequence, global_state, input_mask, aux_loss

    # %%
    def forward(self, sequence, input_mask):
        input_mask = input_mask.unsqueeze(-1)
        sequence = sequence * input_mask

        sequence, global_state, input_mask, aux_loss = self.encoder_block(sequence, input_mask)
        sequence = sequence * input_mask
        return {"sequence": sequence, "global_state": global_state, "input_mask": input_mask, "aux_loss": aux_loss}
