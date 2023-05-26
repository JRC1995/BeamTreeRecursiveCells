import torch as T
import torch.nn as nn
import torch.nn.functional as F
from optimizers import *
from torch.optim import *
from controllers import *


class classifier_agent:
    def __init__(self, model, config, device):
        self.model = model
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = eval(config["optimizer"])

        if "memory_lr" in config:
            grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if "memory_values" in n and p.requires_grad],
                 'weight_decay': config["weight_decay"], 'lr': config["memory_lr"]},
                {'params': [p for n, p in model.named_parameters() if
                            "memory_values" not in n and p.requires_grad],
                 'weight_decay': config["weight_decay"], 'lr': config["lr"]}]
            self.optimizer = optimizer(grouped_parameters,  # optimizer_grouped_parameters,
                                       lr=config["lr"],
                                       weight_decay=config["weight_decay"])

            if config["different_betas"]:
                self.optimizer = optimizer(grouped_parameters,
                                           lr=config["lr"],
                                           weight_decay=config["weight_decay"],
                                           betas=(0, 0.999),
                                           eps=1e-9)
            else:
                self.optimizer = optimizer(grouped_parameters,  # optimizer_grouped_parameters,
                                           lr=config["lr"],
                                           weight_decay=config["weight_decay"])

        else:
            if config["different_betas"]:
                self.optimizer = optimizer(self.parameters,
                                           lr=config["lr"],
                                           weight_decay=config["weight_decay"],
                                           betas=(0, 0.999),
                                           eps=1e-9)
            else:
                self.optimizer = optimizer(self.parameters,  # optimizer_grouped_parameters,
                                           lr=config["lr"],
                                           weight_decay=config["weight_decay"])

        self.scheduler, self.epoch_level_scheduler = get_scheduler(config, self.optimizer)
        self.config = config
        self.device = device
        self.DataParallel = config["DataParallel"]

        self.optimizer.zero_grad()

        """
        if "beam" in self.config["model_name"].lower():
            if self.config["classes_num"] == 2:
                self.criterion = nn.BCELoss(reduction='mean')
        else:
        """
        if self.config["classes_num"] > 2:
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # %%
    def loss_fn(self, logits, labels, train=False, aux_loss=None):

        if self.config["classes_num"] == 2:
            labels = F.one_hot(labels, num_classes=2).float()
            labels = labels[..., 1].unsqueeze(-1)

        loss = self.criterion(logits, labels)
        if aux_loss is not None and train:
            loss = loss + aux_loss
        return loss

    # %%
    def beam_loss_fn(self, logits, labels, train=False, aux_loss=None):

        labels = F.one_hot(labels, num_classes=self.config["classes_num"]).float()
        N, C = labels.size()

        if self.config["classes_num"] == 2:
            labels = labels[..., 1].unsqueeze(-1)
            assert labels.size() == logits.size()
            loss = self.criterion(logits, labels)
        else:
            assert labels.size() == logits.size()
            loss = -T.sum(labels * T.log(logits + 1e-8), dim=-1)
            loss = loss.mean()

        if aux_loss is not None and train:
            loss = loss + aux_loss
        return loss

    # %%
    def run(self, batch, train=True):

        if train:
            self.model = self.model.train()
        else:
            self.model = self.model.eval()

        if not self.DataParallel:
            batch["sequences_vec"] = batch["sequences_vec"].to(self.device)
            batch["labels"] = batch["labels"].to(self.device)
            batch["input_masks"] = batch["input_masks"].to(self.device)
            if "parse_trees" in batch:
                batch["parse_trees"] = batch["parse_trees"].to(self.device)

        N = batch["sequences_vec"].size(0)
        output_dict = self.model(batch)
        logits = output_dict["logits"]
        labels = batch["labels"].to(logits.device)
        aux_loss = output_dict["aux_loss"]

        """
        if "beam" in self.config["model_name"].lower():
            beam_size = self.config["beam_size"]
            classes_num = self.config["classes_num"] if self.config["classes_num"] > 2 else 1
            logits = logits.view(N, beam_size, classes_num)
            beam_scores = output_dict["beam_scores"]
            assert beam_scores.size() == (N, beam_size, 1)
            if self.config["classes_num"] == 2:
                logits = T.sigmoid(logits)
            else:
                logits = T.softmax(logits, dim=-1)
            #print("logits: ", logits[-3])
            #print("beam_scores: ", beam_scores[-3])
            logits = T.sum(beam_scores * logits, dim=1)
            #print("marginalized logits: ", logits[-3])
            assert logits.size() == (N, classes_num)

            loss = self.beam_loss_fn(logits=logits, labels=labels,
                                     train=train, aux_loss=aux_loss)

            if self.config["classes_num"] == 2:
                predictions = T.where(logits >= 0.5,
                                      T.ones_like(logits).int().to(logits.device),
                                      T.zeros_like(logits).int().to(logits.device))
                predictions = predictions.squeeze(-1)
            else:
                predictions = T.argmax(logits, dim=-1)
        else:
        """
        loss = self.loss_fn(logits=logits, labels=labels,
                            train=train, aux_loss=aux_loss)

        if self.config["classes_num"] == 2:
            predictions = T.where(T.sigmoid(logits) >= 0.5,
                                  T.ones_like(logits).int().to(logits.device),
                                  T.zeros_like(logits).int().to(logits.device))
            predictions = predictions.squeeze(-1)
        else:
            predictions = T.argmax(logits, dim=-1)
        predictions = predictions.detach().cpu().numpy().tolist()

        labels = batch["labels"].cpu().numpy().tolist()
        metrics = self.eval_fn(predictions, labels)
        metrics["loss"] = loss.item()

        items = {"display_items": {"sequences": batch["sequences"],
                                   "predictions": predictions,
                                   "labels": labels},
                 "loss": loss,
                 "metrics": metrics}

        return items

    # %%
    def backward(self, loss):
        loss.backward()

    # %%
    def step(self):
        if self.config["max_grad_norm"] is not None:
            T.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
        self.optimizer.step()
        self.optimizer.zero_grad()
        if (not self.epoch_level_scheduler) and self.scheduler is not None:
            self.scheduler.step()

        self.config["current_lr"] = self.optimizer.param_groups[-1]["lr"]

    # %%
    def eval_fn(self, predictions, labels):
        correct_prediction_list = [1 if prediction == label else 0 for prediction, label in zip(predictions, labels)]
        correct_predictions = sum(correct_prediction_list)
        total = len(correct_prediction_list)

        accuracy = correct_predictions / total if total > 0 else 0
        accuracy *= 100

        return {"correct_predictions": correct_predictions,
                "total": total,
                "accuracy": accuracy}
