import torch as T
import torch.nn as nn
import torch.nn.functional as F
import math


class CYKCell(nn.Module):
    def __init__(self, config):
        super(CYKCell, self).__init__()

        self.hidden_size = config["hidden_size"]
        self.in_dropout = config["in_dropout"]
        self.hidden_dropout = config["dropout"]
        self.cell_hidden_size = 4 * config["hidden_size"]

        self.scorer = nn.Linear(self.hidden_size, 1)

        self.initial_transform_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.wcell1 = nn.Linear(2 * self.hidden_size, self.cell_hidden_size)
        self.wcell2 = nn.Linear(self.cell_hidden_size, 4 * self.hidden_size)
        self.LN1 = nn.LayerNorm(self.hidden_size)
        self.LN2 = nn.LayerNorm(self.hidden_size)

        self.eps = 1e-8

    # %%
    def sum_normalize(self, logits, dim=-1):
        return logits / T.sum(logits + self.eps, keepdim=True, dim=dim)

    # %%
    def initial_transform(self, sequence):
        sequence = self.LN1(self.initial_transform_layer(sequence))
        return sequence

    # %%
    def composer(self, child1, child2):
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
    def encoder_block(self, sequence, input_mask):
        N, S, D = sequence.size()
        """
        Initial Transform
        """
        sequence = self.initial_transform(sequence)
        sequence = sequence * input_mask

        chart = [sequence]
        chart_mask = [input_mask]

        for row in range(1, S):
            left_stack = []
            right_stack = []
            left_mask_stack = []
            right_mask_stack = []
            for j in range(row):
                left = chart[j][:, 0:S - row, :]
                right = chart[row - j - 1][:, j + 1:, :]

                left_mask = chart_mask[j][:, 0:S-row, :]
                right_mask = chart_mask[row - j - 1][:, j + 1:, :]

                #print("leftsize: ", left.size())
                #print("S: ", S)
                #print("row: ", row)


                assert left.size() == (N, S - row, self.hidden_size)
                assert right.size() == (N, S - row, self.hidden_size)
                assert left_mask.size() == (N, S - row, 1)
                assert right_mask.size() == (N, S - row, 1)

                left_stack.append(left)
                right_stack.append(right)
                left_mask_stack.append(left_mask)
                right_mask_stack.append(right_mask)

            left_stack = T.stack(left_stack, dim=1)
            right_stack = T.stack(right_stack, dim=1)
            left_mask_stack = T.stack(left_mask_stack, dim=1)
            right_mask_stack = T.stack(right_mask_stack, dim=1)
            assert left_stack.size() == (N, row, S - row, self.hidden_size)
            assert right_stack.size() == (N, row, S - row, self.hidden_size)
            assert left_mask_stack.size() == (N, row, S - row, 1)
            assert right_mask_stack.size() == (N, row, S - row, 1)

            left_stack = left_stack.view(N * row, S - row, self.hidden_size)
            right_stack = right_stack.view(N * row, S - row, self.hidden_size)

            combined_stack = self.composer(left_stack, right_stack)

            combined_stack = combined_stack.view(N, row, S - row, self.hidden_size)
            left_stack = left_stack.view(N, row, S-row, self.hidden_size)
            assert combined_stack.size() == (N, row, S - row, self.hidden_size)
            combined_stack = combined_stack / (T.norm(combined_stack, keepdim=True, dim=-1) + self.eps)
            combined_stack = right_mask_stack * combined_stack + (1 - right_mask_stack) * left_stack
            assert combined_stack.size() == (N, row, S - row, self.hidden_size)
            #assert self.scorer(combined_stack).size() == (N, row, S - row, 1)
            combined_scores = F.softmax(self.scorer(combined_stack), dim=1)
            assert combined_scores.size() == (N, row, S - row, 1)

            combined_scores = self.sum_normalize(right_mask_stack * combined_scores, dim=1)

            new_row = T.sum(combined_scores * combined_stack, dim=1)
            assert new_row.size() == (N, S - row, self.hidden_size)

            new_mask = T.sum(combined_scores * right_mask_stack, dim=1)

            chart.append(new_row)
            chart_mask.append(new_mask)

        lengths = T.sum(input_mask.squeeze(-1), dim=1).detach().cpu().numpy().tolist()
        global_state = []
        for b, j in enumerate(lengths):
            j = int(j)
            global_state.append(chart[j - 1][b, 0, :])
        global_state = T.stack(global_state, dim=0)

        assert global_state.size() == (N, self.hidden_size)

        penalty = None

        return sequence, global_state, penalty

    # %%
    def forward(self, sequence, input_mask):
        input_mask = input_mask.unsqueeze(-1)
        sequence = sequence * input_mask

        sequence, global_state, aux_loss = self.encoder_block(sequence, input_mask)
        sequence = sequence * input_mask
        return {"sequence": sequence, "global_state": global_state, "input_mask": input_mask, "aux_loss": aux_loss}
