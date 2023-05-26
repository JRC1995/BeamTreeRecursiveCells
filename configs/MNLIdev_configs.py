class optimizer_config:
    def __init__(self):
        # optimizer config
        self.max_grad_norm = 5
        self.batch_size = 128
        self.train_batch_size = 128
        self.dev_batch_size = 128
        self.bucket_size_factor = 10
        self.DataParallel = False
        self.num_workers = 6
        self.weight_decay = 1e-2
        self.lr = 1e-3
        self.epochs = 50
        self.early_stop_patience = 4
        self.scheduler = "ReduceLROnPlateau"
        self.scheduler_patience = 2
        self.scheduler_reduce_factor = 0.5
        self.optimizer = "Ranger"
        self.save_by = "accuracy"
        self.metric_direction = 1
        self.different_betas = False
        self.chunk_size = -1
        self.display_metric = "accuracy"
        self.greedy_training = False


class base_config(optimizer_config):
    def __init__(self):
        super().__init__()
        self.word_embd_freeze = True
        self.initial_transform = False
        self.batch_pair = True
        self.parse_trees = False
        self.embd_dim = 300
        self.input_size = 300
        self.hidden_size = 300
        self.rao_k = 10
        self.classifier_hidden_size = 300
        self.rao = False
        self.stochastic = False
        self.test_time_stochastic = False
        self.treedrop = False
        self.global_state_only = True
        self.global_state_return = True
        self.gumbel_diff = False


class GumbelTreeLSTM_config(base_config):
    def __init__(self):
        super().__init__()
        self.in_dropout = 0.4
        self.dropout = 0.1
        self.out_dropout = 0.1
        self.conv_decision = False
        self.encoder_type = "GumbelTreeLSTM"
        self.model_name = "(GumbelTreeLSTM)"


class RCell_config(base_config):
    def __init__(self):
        super().__init__()
        self.in_dropout = 0.4
        self.dropout = 0.1
        self.out_dropout = 0.1
        self.bidirectional = False
        self.encoder_type = "BiCell"
        self.model_name = "(RCell)"


class BalancedTreeCell_config(RCell_config):
    def __init__(self):
        super().__init__()
        self.encoder_type = "BalancedTreeCell"
        self.model_name = "(BalancedTreeCell)"

class CRvNN_config(RCell_config):
    def __init__(self):
        super().__init__()
        self.train_batch_size = 64
        self.encoder_type = "CRvNN"
        self.model_name = "(CRvNN)"

class OrderedMemory_config(RCell_config):
    def __init__(self):
        super().__init__()
        self.batch_pair = True
        self.dropout = 0.1
        self.memory_dropout = 0.1
        self.in_dropout = 0.4
        self.out_dropout = 0.1
        self.memory_slots = 12
        self.hidden_size = 300
        self.encoder_type = "OrderedMemory"
        self.model_name = "(ordered_memory)"


class GumbelTreeCell_config(RCell_config):
    def __init__(self):
        super().__init__()
        self.conv_decision = False
        self.encoder_type = "GumbelTreeCell"
        self.model_name = "(GumbelTreeCell)"


class RandomTreeCell_config(RCell_config):
    def __init__(self):
        super().__init__()
        self.conv_decision = False
        self.encoder_type = "RandomTreeCell"
        self.model_name = "(RandomTreeCell)"


class MCGumbelTreeCell_config(RCell_config):
    def __init__(self):
        super().__init__()
        self.conv_decision = False
        self.sample_size = 5
        self.encoder_type = "MCGumbelTreeCell"
        self.model_name = "(MCGumbelTreeCell)"


class BeamTreeLSTM_config(GumbelTreeLSTM_config):
    def __init__(self):
        super().__init__()
        self.conv_decision = False
        self.diffop1 = False
        self.diffop2 = False
        self.stochastic = True
        self.beam_size = 5
        self.encoder_type = "BeamGumbelTreeLSTM"
        self.model_name = "(BeamTreeLSTM)"

class BeamTreeCell_config(GumbelTreeCell_config):
    def __init__(self):
        super().__init__()
        self.train_batch_size = 64
        self.conv_decision = False
        self.diffop1 = False
        self.diffop2 = False
        self.stochastic = True
        self.test_time_stochastic = False
        self.beam_size = 5
        self.encoder_type = "BeamGumbelTreeCell"
        self.model_name = "(BeamTreeCell)"

class SmallerBeamTreeCell_config(BeamTreeCell_config):
    def __init__(self):
        super().__init__()
        self.beam_size = 2
        self.train_batch_size = 128
        self.encoder_type = "BeamGumbelTreeCell"
        self.model_name = "(SmallerBeamTreeCell)"

class DiffBeamTreeCell_config(BeamTreeCell_config):
    def __init__(self):
        super().__init__()
        self.train_batch_size = 32
        self.encoder_type = "DiffBeamTreeCell"
        self.model_name = "(DiffBeamTreeCell)"


class SmallerDiffBeamTreeCell_config(DiffBeamTreeCell_config):
    def __init__(self):
        super().__init__()
        self.beam_size = 2
        self.train_batch_size = 64
        self.encoder_type = "DiffBeamTreeCell"
        self.model_name = "(SmallerDiffBeamTreeCell)"


