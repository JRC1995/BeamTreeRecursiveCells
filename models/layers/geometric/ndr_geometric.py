import torch
import torch.nn
from models.layers.geometric.layer_with_visualization import LayerWithVisualization
from models.layers.geometric.regularized_layer import RegularizedLayer
from typing import Dict, Any
import framework
from models.layers.geometric.direction_sensitive_geometric import DirectionSensitiveGeometricAttentionMyInit, AttentionMask


class NDRGeometric(RegularizedLayer, LayerWithVisualization):
    def __init__(self, config, **kwargs):
        super().__init__()
        d_model = config["hidden_size"]
        nhead = config["heads"]
        dropout = config["dropout"]
        scalar_gate = False
        attention_dropout = config["dropout"]
        dim_feedforward = config["ff_dim"]
        global_content_bias = True
        normalize_score = True
        p_gate_drop = 0.05
        gate_size_multiplier = 1

        self.plot_cache = []

        self.reg_loss = 0

        dim_feedforward = dim_feedforward or (4 * d_model)
        self.att = DirectionSensitiveGeometricAttentionMyInit(d_model, nhead, dropout=attention_dropout,
                                                              normalize_score=normalize_score,
                                                              global_content_bias=global_content_bias)

        self.p1 = torch.nn.Linear(d_model, dim_feedforward)
        self.p2 = torch.nn.Linear(dim_feedforward, d_model)

        self.g1 = torch.nn.Linear(d_model, d_model * gate_size_multiplier)
        self.g2 = torch.nn.Linear(d_model * gate_size_multiplier, 1 if scalar_gate else d_model)

        self.nmerge = torch.nn.LayerNorm(d_model)
        self.no = torch.nn.LayerNorm(d_model)

        self.drop = torch.nn.Dropout(dropout)

        self.g2.bias.data.fill_(-3)
        self.p_gate_drop = p_gate_drop

        self.reset_parameters()

    def forward(self, sequence, input_mask):
        src = sequence
        N, S, D = src.size()
        assert input_mask.size() == (N, S)

        mask = AttentionMask(~(input_mask.bool()), None)
        #mask.src_length_mask =

        input = self.att(src, src, mask)
        net = self.nmerge(src + self.drop(input))

        mid = self.drop(torch.relu(self.p1(net)))
        proj = self.p2(mid)
        proj = self.no(proj)

        gate = self.g2(self.drop(torch.relu(self.g1(net))))
        bgate = torch.sigmoid(gate)

        if self.training and self.p_gate_drop > 0:
            bgate = bgate.masked_fill(
                torch.rand(*bgate.shape[:-1], 1, device=bgate.device, dtype=bgate.dtype) < self.p_gate_drop, 0)

        if self.visualization_enabled:
            self.plot_cache.append(bgate[0])

        src = src * (1 - bgate) + proj * bgate

        return src

    def plot(self, options: Dict[str, Any]) -> Dict[str, Any]:
        r = {}
        if self.visualization_enabled:
            r["gate"] = framework.visualize.plot.AnimatedHeatmap(
                torch.stack(self.plot_cache, 0).transpose(1, 2),
                ylabel="dest", xlabel="src", textval=False, x_marks=options.get("steplabel"))
            self.plot_cache.clear()

        return r

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.p1.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.p2.weight, gain=torch.nn.init.calculate_gain('tanh'))

        torch.nn.init.xavier_uniform_(self.g1.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.g2.weight, gain=torch.nn.init.calculate_gain('sigmoid'))
