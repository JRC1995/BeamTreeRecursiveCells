class optimizer_config:
    def __init__(self):
        # optimizer config
        self.max_grad_norm = 1
        self.batch_size = 128
        self.train_batch_size = 128
        self.dev_batch_size = 128
        self.bucket_size_factor = 5
        self.DataParallel = False
        self.num_workers = 6
        self.weight_decay = 1e-2
        self.lr = 1e-3
        self.epochs = 100
        self.early_stop_patience = 4
        self.scheduler = "ReduceLROnPlateau"
        self.scheduler_patience = 2
        self.scheduler_reduce_factor = 0.5
        self.optimizer = "Ranger"
        self.save_by = "loss"
        self.metric_direction = -1
        self.different_betas = False
        self.chunk_size = -1
        self.display_metric = "accuracy"


class base_config(optimizer_config):
    def __init__(self):
        super().__init__()
        self.word_embd_freeze = False
        self.initial_transform = False
        self.batch_pair = False
        self.embd_dim = 128
        self.input_size = 128
        self.hidden_size = 128
        self.rao_k = 10
        self.classifier_hidden_size = 128
        self.rao = False
        self.stochastic = False
        self.test_time_stochastic = False
        self.treedrop = False
        self.global_state_only = True
        self.global_state_return = True
        self.gumbel_diff = False
        self.parse_trees = False


class NDR_config(base_config):
    def __init__(self):
        super().__init__()
        self.input_size = 512
        self.hidden_size = 512
        self.embd_dim = 512
        self.batch_pair = False
        self.batch_size = 512
        self.train_batch_size = 128
        self.optimizer = "AdamW"
        self.weight_decay = 0.09
        self.in_dropout = 0.1
        self.scheduler = None
        self.early_stop_patience = 100
        self.save_by = "accuracy"
        self.dropout = 0.1
        self.out_dropout = 0.1
        self.train_max_depth = 20
        self.test_max_depth = 24
        self.metric_direction = +1
        self.lr = 2e-4
        self.ff_dim = 1024
        self.heads = 16
        self.encoder_type = "ndr_geometric_stack"
        self.model_name = "(NDR)"


class GumbelTreeLSTM_config(base_config):
    def __init__(self):
        super().__init__()
        self.in_dropout = 0.1
        self.dropout = 0.1
        self.out_dropout = 0.1
        self.conv_decision = False
        self.encoder_type = "GumbelTreeLSTM"
        self.model_name = "(GumbelTreeLSTM)"


class BiCell_config(base_config):
    def __init__(self):
        super().__init__()
        self.in_dropout = 0.1
        self.dropout = 0.1
        self.out_dropout = 0.1
        self.bidirectional = True
        self.encoder_type = "BiCell"
        self.model_name = "(BiCell)"


class GoldTreeCell_config(BiCell_config):
    def __init__(self):
        super().__init__()
        self.parse_trees = True
        self.encoder_type = "GoldTreeCell"
        self.model_name = "(GoldTreeCell)"


class RCell_config(base_config):
    def __init__(self):
        super().__init__()
        self.in_dropout = 0.1
        self.dropout = 0.1
        self.out_dropout = 0.1
        self.bidirectional = False
        self.encoder_type = "BiCell"
        self.model_name = "(RCell)"

class RandomTreeCell_config(BiCell_config):
    def __init__(self):
        super().__init__()
        self.conv_decision = False
        self.encoder_type = "RandomTreeCell"
        self.model_name = "(RandomTreeCell)"

class BSRPCell_config(base_config):
    def __init__(self):
        super().__init__()
        self.in_dropout = 0.1
        self.dropout = 0.1
        self.out_dropout = 0.1
        self.beam_size = 5
        self.encoder_type = "BSRPCell"
        self.model_name = "(BSRPCell)"

class BigBSRPCell_config(base_config):
    def __init__(self):
        super().__init__()
        self.in_dropout = 0.1
        self.dropout = 0.1
        self.out_dropout = 0.1
        self.beam_size = 8
        self.encoder_type = "BSRPCell"
        self.model_name = "(BigBSRPCell)"

class CYKCell_config(base_config):
    def __init__(self):
        super().__init__()
        self.train_batch_size = 16
        self.dev_batch_size = 1
        self.in_dropout = 0.1
        self.dropout = 0.1
        self.out_dropout = 0.1
        self.encoder_type = "CYKCell"
        self.model_name = "(CYKCell)"

class CRvNN_config(BiCell_config):
    def __init__(self):
        super().__init__()
        self.encoder_type = "CRvNN"
        self.model_name = "(CRvNN)"

class OrderedMemory_config(BiCell_config):
    def __init__(self):
        super().__init__()
        self.dropout = 0.1
        self.memory_dropout = 0.1
        self.in_dropout = 0.1
        self.out_dropout = 0.1
        self.memory_slots = 21
        self.encoder_type = "OrderedMemory"
        self.model_name = "(ordered_memory)"


class BalancedTreeCell_config(BiCell_config):
    def __init__(self):
        super().__init__()
        self.encoder_type = "BalancedTreeCell"
        self.model_name = "(BalancedTreeCell)"


class GumbelTreeCell_config(BiCell_config):
    def __init__(self):
        super().__init__()
        self.conv_decision = False
        self.encoder_type = "GumbelTreeCell"
        self.model_name = "(GumbelTreeCell)"


class MCGumbelTreeCell_config(BiCell_config):
    def __init__(self):
        super().__init__()
        self.conv_decision = False
        self.sample_size = 5
        self.encoder_type = "MCGumbelTreeCell"
        self.model_name = "(MCGumbelTreeCell)"


class BeamTreeLSTM_config(GumbelTreeCell_config):
    def __init__(self):
        super().__init__()
        self.conv_decision = False
        self.diffop1 = False
        self.diffop2 = False
        self.stochastic = True
        self.beam_size = 5
        self.encoder_type = "BeamGumbelTreeLSTM"
        self.model_name = "(BeamTreeLSTM)"


class DiffBeamTreeCell_config(GumbelTreeCell_config):
    def __init__(self):
        super().__init__()
        self.train_batch_size = 64
        self.conv_decision = False
        self.diffop1 = False
        self.diffop2 = False
        self.stochastic = True
        self.beam_size = 5
        self.encoder_type = "DiffBeamTreeCell"
        self.model_name = "(DiffBeamTreeCell)"


class SmallerDiffBeamTreeCell_config(GumbelTreeCell_config):
    def __init__(self):
        super().__init__()
        self.conv_decision = False
        self.diffop1 = False
        self.diffop2 = False
        self.stochastic = True
        self.beam_size = 2
        self.encoder_type = "DiffBeamTreeCell"
        self.model_name = "(SmallerDiffBeamTreeCell)"

class BeamTreeCell_config(GumbelTreeCell_config):
    def __init__(self):
        super().__init__()
        self.conv_decision = False
        self.diffop1 = False
        self.diffop2 = False
        self.stochastic = True
        self.test_time_stochastic = False
        self.beam_size = 5
        self.encoder_type = "BeamGumbelTreeCell"
        self.model_name = "(BeamTreeCell)"

class SmallerBeamTreeCell_config(GumbelTreeCell_config):
    def __init__(self):
        super().__init__()
        self.conv_decision = False
        self.diffop1 = False
        self.diffop2 = False
        self.stochastic = True
        self.test_time_stochastic = False
        self.beam_size = 2
        self.encoder_type = "BeamGumbelTreeCell"
        self.model_name = "(SmallerBeamTreeCell)"
