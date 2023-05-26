import copy
import random
import zlib
from pathlib import Path

import numpy as np
import torch as T
import torch.nn as nn

from collaters import *
from configs.configLoader import load_config
from controllers.metric_controller import metric_fn, compose_dev_metric
from argparser import get_args
from trainers import Trainer
from utils.checkpoint_utils import load_temp_checkpoint, load_infer_checkpoint, save_infer_checkpoint, \
    save_temp_checkpoint
from utils.data_utils import load_data, load_dataloaders
from utils.display_utils import example_display_fn, step_display_fn, display
from utils.param_utils import param_display_fn, param_count
from utils.path_utils import load_paths
from models import *
from agents import *
import numpy as np


def run(args, config, time=0):
    device = T.device(args.device)
    config["device"] = device

    display_string = "Parsed Arguments: {}\n\n".format(args)
    display_string += "Configs:\n"
    for k, v in config.items():
        display_string += "{}: {}\n".format(k, v)
    display_string += "\n"

    paths, checkpoint_paths, metadata = load_paths(args, time)
    data, config = load_data(paths, metadata, args, config)

    model = eval("{}_framework".format(args.model_type))
    model = model(data=data,
                  config=config)
    model = model.to(device)

    if config["DataParallel"]:
        model = nn.DataParallel(model)

    if args.display_params:
        display_string += param_display_fn(model)

    total_parameters = param_count(model)
    display_string += "Total Parameters: {}\n\n".format(total_parameters)

    print(display_string)

    agent = eval("{}_agent".format(args.model_type))

    agent = agent(model=model,
                  config=config,
                  device=device)

    agent, epochs_taken = load_infer_checkpoint(agent, checkpoint_paths, paths)
    vocab2idx = data["vocab2idx"]
    idx2vocab = {v: k for k, v in vocab2idx.items()}
    UNK_id = data["UNK_id"]

    texts = ["i did not like a single minute of this film",
             "i shot an elephant in my pajamas",
             "john saw a man with binoculars",
             "roger dodger is one of the most compelling variations of this theme",
             "recursive neural networks can compose sequences according to their underlying hierarchical syntactic structures"]

    # texts = ["roger dodger is one of the most compelling variation of this theme ."]
    for text in texts:
        original_text = copy.deepcopy(text)
        text = text.lower().split(" ")
        text_idx = [idx2vocab.get(word, UNK_id) for word in text]
        input_mask = [1] * len(text_idx)

        batch = {}
        batch["sequences1_vec"] = T.tensor([text_idx]).long().to(device)
        batch["sequences2_vec"] = T.tensor([text_idx]).long().to(device)
        batch["input_masks1"] = T.tensor([input_mask]).float().to(device)
        batch["input_masks2"] = T.tensor([input_mask]).float().to(device)
        batch["temperature"] = None

        agent.model.eval()
        output_dict = agent.model(batch)
        paths = output_dict["paths1"]
        path_scores = output_dict["path_scores1"]
        N, B, S = paths.size()
        N, B = path_scores.size()

        paths = paths.view(B, S).detach().cpu().numpy().tolist()
        path_scores = path_scores.view(B).detach().cpu().numpy().tolist()
        beam_paths = []
        for b in range(B):
            path = paths[b]
            otext = copy.deepcopy(text)

            while len(otext) > 1:
                len_text = len(otext)
                if len_text == 2:
                    otext = ["(" + otext[0] + " " + otext[1] + ")"]
                else:
                    choice_id = np.argmax(path[0:len_text - 1])
                    otext_ = otext[0:choice_id] + ["(" + otext[choice_id] + " " + otext[choice_id + 1] + ")"]
                    if choice_id + 1 < len_text - 1:
                        otext_ = otext_ + otext[choice_id + 2:]
                    otext = otext_
                    path = path[len_text - 1:]
            beam_paths.append(otext[0])

        print("INPUT TEXT: ", original_text)
        print("BEAMS: ")
        for b in range(B):
            print("Score: {}, Path: {}".format(path_scores[b], beam_paths[b]))
        print("\n\n")


if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    config = load_config(args)

    config["encoder_type"] = config["encoder_type"] + "_transparent"
    run(args, config)
