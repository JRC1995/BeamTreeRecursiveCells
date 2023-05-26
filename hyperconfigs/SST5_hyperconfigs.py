class RCell_hyperconfig:
    def __init__(self):
        super().__init__()
        self.dropout = [0.1, 0.2, 0.3, 0.4]
        self.io_dropout = [0.1, 0.2, 0.3, 0.4]
        self.max_trials = 50
        self.allow_repeat = False
        self.hyperalgo = "hyperopt.rand.suggest"
        self.epochs = 20
        self.limit = -1

    def process_config(self, config):
        config["in_dropout"] = config["io_dropout"]
        config["out_dropout"] = config["io_dropout"]
        config["dropout"] = config["dropout"]
        config["early_stop_patience"] = 4
        return config

class GumbelTreeLSTM_hyperconfig:
    def __init__(self):
        super().__init__()
        self.io_dropout = [0.1, 0.2, 0.3, 0.4]
        self.dropout = [0.1, 0.2, 0.3, 0.4]
        self.max_trials = 50
        self.allow_repeat = False
        self.hyperalgo = "hyperopt.rand.suggest"
        self.epochs = 20
        self.limit = -1

    def process_config(self, config):
        config["in_dropout"] = config["io_dropout"]
        config["out_dropout"] = config["io_dropout"]
        config["dropout"] = config["dropout"]
        config["early_stop_patience"] = 4
        return config