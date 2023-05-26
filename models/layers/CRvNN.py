import torch as T
import torch.nn as nn
import torch.nn.functional as F


class CRvNN(nn.Module):
    def __init__(self, config):
        super(CRvNN, self).__init__()

        self.config = config
        self.max_depth = 25 #config["max_depth"]
        self.scorer_window_size = 5 #config["scorer_window_size"]
        self.hidden_size = config["hidden_size"]
        self.cell_hidden_size = 4 * config["hidden_size"]
        self.stop_threshold = 0.1 #config["stop_threshold"]
        self.hidden_dropout = config["dropout"]

        self.START = nn.Parameter(T.randn(self.hidden_size))
        self.END = nn.Parameter(T.randn(self.hidden_size))

        self.conv_layer = nn.Linear(self.scorer_window_size * self.hidden_size, self.hidden_size)
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
    def augment_sequence(self, sequence, input_mask):
        N, S, D = sequence.size()
        assert input_mask.size() == (N, S, 1)

        """
        AUGMENT SEQUENCE WITH START AND END TOKENS
        """
        # ADD START TOKEN
        START = self.START.view(1, 1, D).repeat(N, 1, 1)
        sequence = T.cat([START, sequence], dim=1)
        assert sequence.size() == (N, S + 1, D)
        input_mask = T.cat([T.ones(N, 1, 1).float().to(input_mask.device), input_mask], dim=1)
        assert input_mask.size() == (N, S + 1, 1)

        # ADD END TOKEN
        input_mask_no_end = T.cat([input_mask.clone(), T.zeros(N, 1, 1).float().to(input_mask.device)], dim=1)
        input_mask_yes_end = T.cat([T.ones(N, 1, 1).float().to(input_mask.device), input_mask.clone()], dim=1)
        END_mask = input_mask_yes_end - input_mask_no_end
        assert END_mask.size() == (N, S + 2, 1)

        END = self.END.view(1, 1, D).repeat(N, S + 2, 1)
        sequence = T.cat([sequence, T.zeros(N, 1, D).float().to(sequence.device)], dim=1)
        sequence = END_mask * END + (1 - END_mask) * sequence

        input_mask = input_mask_yes_end
        input_mask_no_start = T.cat([T.zeros(N, 1, 1).float().to(input_mask.device),
                                     input_mask[:, 1:, :]], dim=1)

        return sequence, input_mask, END_mask, input_mask_no_start, input_mask_no_end

    # %%
    """
    def compute_neighbor_probs(self, exist_probs, input_mask, ones_matrix):
        N, S, _ = input_mask.size()
        assert input_mask.size() == (N, S, 1)
        input_mask = input_mask.permute(0, 2, 1).contiguous()
        assert input_mask.size() == (N, 1, S)

        assert exist_probs.size() == (N, S, 1)
        exist_probs = exist_probs.permute(0, 2, 1).contiguous()
        assert exist_probs.size() == (N, 1, S)

        input_mask_flipped = T.flip(input_mask.clone(), dims=[2])
        exist_probs_flipped = T.flip(exist_probs.clone(), dims=[2])

        # (flipped for left retrieval)
        input_mask = T.stack([input_mask_flipped, input_mask], dim=1)
        exist_probs = T.stack([exist_probs_flipped, exist_probs], dim=1)

        assert input_mask.size() == (N, 2, 1, S)
        assert exist_probs.size() == (N, 2, 1, S)

        exist_probs_matrix = exist_probs.repeat(1, 1, S, 1) * input_mask
        assert exist_probs_matrix.size() == (N, 2, S, S)
        right_exist_probs_matrix = T.triu(exist_probs_matrix, diagonal=1)  # mask self and left

        not_exist_yet_probs_matrix = T.cumprod(1 - right_exist_probs_matrix, dim=-1)
        not_exist_yet_probs_matrix = T.cat([ones_matrix, not_exist_yet_probs_matrix[..., 0:-1]], dim=-1)
        assert not_exist_yet_probs_matrix.size() == (N, 2, S, S)

        right_neighbor_probs = not_exist_yet_probs_matrix * right_exist_probs_matrix
        right_neighbor_probs = right_neighbor_probs * input_mask

        left_neighbor_probs = right_neighbor_probs[:, 0, :, :]
        left_neighbor_probs = T.flip(left_neighbor_probs, dims=[1, 2])
        right_neighbor_probs = right_neighbor_probs[:, 1, :, :]

        return left_neighbor_probs, right_neighbor_probs
    """

    def compute_neighbor_probs(self, exist_probs, input_mask, ones_matrix):
        N, S, _ = input_mask.size()
        assert input_mask.size() == (N, S, 1)
        input_mask = input_mask.permute(0, 2, 1).contiguous()
        assert input_mask.size() == (N, 1, S)

        assert exist_probs.size() == (N, S, 1)
        exist_probs = exist_probs.permute(0, 2, 1).contiguous()
        assert exist_probs.size() == (N, 1, S)

        input_mask_flipped = T.flip(input_mask.clone(), dims=[2])
        exist_probs_flipped = T.flip(exist_probs.clone(), dims=[2])

        input_mask = T.stack([input_mask_flipped, input_mask], dim=1)
        exist_probs = T.stack([exist_probs_flipped, exist_probs], dim=1)

        assert input_mask.size() == (N, 2, 1, S)
        assert exist_probs.size() == (N, 2, 1, S)

        exist_probs_matrix = exist_probs.repeat(1, 1, S, 1) * input_mask
        assert exist_probs_matrix.size() == (N, 2, S, S)
        right_probs_matrix = T.triu(exist_probs_matrix, diagonal=1)  # mask self and left

        right_probs_matrix_cumsum = T.cumsum(right_probs_matrix, dim=-1)
        assert right_probs_matrix_cumsum.size() == (N, 2, S, S)
        remainders = 1.0 - right_probs_matrix_cumsum

        remainders_from_left = T.cat([T.ones(N, 2, S, 1).float().to(remainders.device), remainders[:, :, :, 0:-1]],
                                     dim=-1)
        assert remainders_from_left.size() == (N, 2, S, S)

        remainders_from_left = T.max(T.zeros(N, 2, S, 1).float().to(remainders.device), remainders_from_left)
        assert remainders_from_left.size() == (N, 2, S, S)

        right_neighbor_probs = T.where(right_probs_matrix_cumsum > 1.0,
                                       remainders_from_left,
                                       right_probs_matrix)

        right_neighbor_probs = right_neighbor_probs * input_mask

        left_neighbor_probs = right_neighbor_probs[:, 0, :, :]
        left_neighbor_probs = T.flip(left_neighbor_probs, dims=[1, 2])
        right_neighbor_probs = right_neighbor_probs[:, 1, :, :]

        return left_neighbor_probs, right_neighbor_probs

    # %%
    def make_window(self, sequence, left_child_probs, right_child_probs, window_size):

        N, S, D = sequence.size()

        left_children_list = []
        right_children_list = []
        left_children_k = sequence.clone()
        right_children_k = sequence.clone()

        for k in range(window_size // 2):
            left_children_k = T.matmul(left_child_probs, left_children_k)
            left_children_list = [left_children_k.clone()] + left_children_list

            right_children_k = T.matmul(right_child_probs, right_children_k)
            right_children_list = right_children_list + [right_children_k.clone()]

        windowed_sequence = left_children_list + [sequence] + right_children_list
        windowed_sequence = T.stack(windowed_sequence, dim=-2)

        assert windowed_sequence.size() == (N, S, window_size, D)

        return windowed_sequence

    # %%
    def initial_transform(self, sequence):
        sequence = self.LN1(self.initial_transform_layer(sequence))
        return sequence

    # %%
    def score_fn(self, windowed_sequence):
        N, S, W, D = windowed_sequence.size()
        windowed_sequence = windowed_sequence.view(N, S, W * D)

        scores = self.scorer(F.gelu(self.conv_layer(windowed_sequence)))

        transition_scores = scores[:, :, 0].unsqueeze(-1)
        # reduce_probs = T.sigmoid(scores[:,:,1].unsqueeze(-1))
        no_op_scores = T.zeros_like(transition_scores).float().to(transition_scores.device)
        scores = T.cat([transition_scores, no_op_scores], dim=-1)
        max_score = T.max(scores)
        exp_scores = T.exp(scores-max_score)

        return exp_scores

    # %%
    def compose(self, child1, child2):
        N, S, D = child1.size()

        concated = T.cat([child1, child2], dim=-1)
        assert concated.size() == (N, S, 2 * D)

        intermediate = F.gelu(self.wcell1(concated))
        intermediate = F.dropout(intermediate, p=self.hidden_dropout, training=self.training)
        contents = self.wcell2(intermediate)
        contents = contents.view(N, S, 4, D)
        gates = T.sigmoid(contents[:, :, 0:3, :])
        parent = contents[:, :, 3, :]
        f1 = gates[..., 0, :]
        f2 = gates[..., 1, :]
        i = gates[..., 2, :]

        transition = self.LN2(f1 * child1 + f2 * child2 + i * parent)

        return transition

    # %%
    def encoder_block(self, sequence, input_mask):

        sequence, input_mask, END_mask, \
        input_mask_no_start, input_mask_no_end = self.augment_sequence(sequence, input_mask)

        N, S, D = sequence.size()

        """
        Initial Preparations
        """
        exist_probs = T.ones(N, S, 1).float().to(sequence.device) * input_mask
        ones_matrix_for_neighbors = T.ones(N, 2, S, 1).float().to(exist_probs.device)
        zeros_token = T.zeros(N, 1, 1).float().to(sequence.device)
        last_mask = T.cat([END_mask[:, 1:, :], zeros_token], dim=1)
        halt_ones = T.ones(N).float().to(sequence.device)
        halt_zeros = T.zeros(N).float().to(sequence.device)
        update_mask = T.ones(N).float().to(sequence.device)
        start_end_last_mask = input_mask_no_start * input_mask_no_end * (1 - last_mask)
        sequence = sequence * input_mask

        """
        Initial Transform
        """
        sequence = self.initial_transform(sequence)
        sequence = sequence * input_mask
        """
        Start Recursion
        """
        t = 0
        U = S-2
        while t < U:

            """
            Backup if needs to be kept unmodified
            """
            previous_sequence = sequence.clone()
            previous_exist_probs = exist_probs.clone()

            """
            Compute Neighbor Retriever Matrices
            """
            left_neighbor_probs, right_neighbor_probs \
                = self.compute_neighbor_probs(exist_probs=exist_probs.clone(),
                                              input_mask=input_mask.clone(),
                                              ones_matrix=ones_matrix_for_neighbors)

            """
            Compute Composition Probabilities
            """
            windowed_sequence = self.make_window(sequence=sequence,
                                                 left_child_probs=left_neighbor_probs,
                                                 right_child_probs=right_neighbor_probs,
                                                 window_size=self.scorer_window_size)

            exp_scores = self.score_fn(windowed_sequence)
            exp_compose_scores = exp_scores[:, :, 0].unsqueeze(-1)
            exp_no_op_scores = exp_scores[:, :, 1].unsqueeze(-1)

            exp_compose_scores = exp_compose_scores * start_end_last_mask

            exp_left_compose_scores = T.matmul(left_neighbor_probs, exp_compose_scores)
            exp_right_compose_scores = T.matmul(right_neighbor_probs, exp_compose_scores)

            exp_scores = T.cat([exp_compose_scores,
                                exp_no_op_scores,
                                exp_left_compose_scores,
                                exp_right_compose_scores], dim=-1)

            normalized_scores = self.sum_normalize(exp_scores, dim=-1)
            compose_scores = normalized_scores[:, :, 0].unsqueeze(-1)
            compose_scores = compose_scores * start_end_last_mask

            # print("t: ", t)
            # print("exist probs: ", exist_probs[-3].squeeze(-1))
            # print("right scores: ", right_scores[-3].squeeze(-1))
            # print("right availibility scores: ", right_availibility_scores[-3].squeeze(-1))
            # print("compose scores: ", compose_scores[-3].squeeze(-1))

            """
            Compute compositions
            """
            left_sequence = windowed_sequence[:, :, self.scorer_window_size // 2 - 1, :]
            compositions = self.compose(child1=left_sequence, child2=sequence)

            """
            UPDATE
            """
            left_compose_scores = T.matmul(left_neighbor_probs, compose_scores)
            sequence = (left_compose_scores * compositions) + ((1 - left_compose_scores) * previous_sequence)
            sequence = sequence * input_mask
            exist_probs = exist_probs * (1.0 - compose_scores) * input_mask

            """
            DYNAMIC HALT
            """
            exist_probs = T.where(update_mask.view(N, 1, 1).expand(N, S, 1) == 1.0,
                                  exist_probs,
                                  previous_exist_probs)

            sequence = T.where(update_mask.view(N, 1, 1).expand(N, S, D) == 1.0,
                               sequence,
                               previous_sequence)

            t += 1
            discrete_exist_probs = T.where(exist_probs > self.stop_threshold,
                                           T.ones_like(exist_probs).to(exist_probs.device),
                                           T.zeros_like(exist_probs).to(exist_probs.device))

            halt_condition_component = T.sum(discrete_exist_probs.squeeze(-1), dim=1) - 2.0
            update_mask = T.where((halt_condition_component <= 1) | (T.sum(input_mask.squeeze(-1), dim=-1) - 2.0 < t),
                                  halt_zeros,
                                  halt_ones)

            if T.sum(update_mask) == 0.0:
                break


        global_state = T.sum(sequence * last_mask, dim=1)
        #aux_loss1 = existential_loss / (invalid_steps + self.eps)

        # REMOVE START AND END TOKENS
        sequence = sequence * (1 - END_mask)
        sequence = sequence[:, 1:-1, :]
        input_mask = input_mask * (1 - END_mask)
        input_mask = input_mask[:, 1:-1, :]

        assert exist_probs.size() == (N, S, 1)
        assert exist_probs.size() == last_mask.size()

        aux_loss = None

        return sequence, global_state, input_mask, aux_loss

    # %%
    def forward(self, sequence, input_mask):

        input_mask = input_mask.unsqueeze(-1)
        sequence = sequence * input_mask

        sequence, global_state, input_mask, aux_loss = self.encoder_block(sequence, input_mask)
        sequence = sequence * input_mask
        return {"sequence": sequence, "global_state": global_state, "input_mask": input_mask, "aux_loss": aux_loss}
