import torch.nn as nn
import torch.nn.functional as F
from models.layers.geometric.ndr_geometric import NDRGeometric
import torch as T


class ndr_geometric_stack(nn.Module):
    def __init__(self, config):
        super(ndr_geometric_stack, self).__init__()

        self.hidden_size = config["hidden_size"]
        self.dropout = config["dropout"]
        self.config = config
        self.train_max_depth = config["train_max_depth"]
        self.test_max_depth = config["test_max_depth"]
        self.START = nn.Parameter(T.randn(self.hidden_size))
        self.END = nn.Parameter(T.randn(self.hidden_size))
        self.EncoderStack = NDRGeometric(config=config)


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
    def forward(self, sequence, input_mask):
        """
        N = Batch Size
        S = Sequence Size
        """
        N = sequence.size(0)

        sequence, input_mask, \
        END_mask, input_mask_no_start, input_mask_no_end = self.augment_sequence(sequence, input_mask.view(N, -1, 1))

        N, S, D = sequence.size()
        input_mask = input_mask.view(N, S)

        sequence = F.dropout(sequence, p=self.dropout, training=self.training)

        if self.training:
            L = self.train_max_depth
        else:
            L = self.test_max_depth

        penalty = None
        for t in range(L):
            sequence = self.EncoderStack(sequence=sequence,
                                         input_mask=input_mask)

        global_state = T.sum(sequence * END_mask, dim=1)


        sequence = sequence * (1-END_mask)
        sequence = sequence[:, 1:-1, :]

        input_mask = input_mask.view(N, S, 1)
        input_mask = input_mask * (1-END_mask)
        input_mask = input_mask[:, 1:-1, :]

        return {"global_state": global_state, "sequence": sequence,
                "input_mask": input_mask, "aux_loss": None}
