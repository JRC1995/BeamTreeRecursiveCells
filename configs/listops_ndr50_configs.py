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
        self.train_batch_size = 512
        self.dev_batch_size = 1
        self.epochs = 50
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
