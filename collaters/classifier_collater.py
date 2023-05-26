import torch as T
import numpy as np
import random
import copy

class classifier_collater:
    def __init__(self, PAD, config, train):
        self.PAD = PAD
        self.config = config
        self.train = train

    def pad(self, items, PAD):
        max_len = max([len(item) for item in items])

        padded_items = []
        item_masks = []
        for item in items:
            mask = [1] * len(item)
            while len(item) < max_len:
                item.append(PAD)
                mask.append(0)
            padded_items.append(item)
            item_masks.append(mask)

        return padded_items, item_masks

    def sort_list_by_idx(self, objs, idx):
        return [objs[i] for i in idx]

    def parse_trees(self, batch_x):

        batch_parse_decisions = []
        max_len = max([len(x) for x in batch_x])

        for x in batch_x:
            parse_decisions = []
            x_ = x
            for i in range(len(x)):
                if len(x_) > 2:
                    left_x = x_[0:-1]
                    right_x = x_[1:]
                    parse_decisions_i = [0] * max_len
                    x_ = []
                    flag = 0
                    for j in range(len(left_x)):
                        if flag == 0:
                            if "[" in left_x[j] and "[" not in right_x[j]:
                                parse_decisions_i[j] = 1
                                if right_x[j] == "]":
                                    l = "b"
                                else:
                                    l = "["
                                x_.append(l)
                                flag = 1
                            else:
                                x_.append(left_x[j])
                        else:
                            x_.append(right_x[j])
                    parse_decisions.append(parse_decisions_i)

            while len(parse_decisions) < max_len:
                parse_decisions.append([0] * max_len)
            batch_parse_decisions.append(parse_decisions)

        return batch_parse_decisions

    def collate_fn(self, batch):
        copy_batch = copy.deepcopy(batch)
        sequences_vec = [obj['sequence_vec'] for obj in copy_batch]
        sequences = [obj['sequence'] for obj in copy_batch]
        labels = [obj['label'] for obj in copy_batch]

        bucket_size = len(sequences_vec)
        if self.train:
            batch_size = self.config["train_batch_size"]
        else:
            batch_size = self.config["dev_batch_size"]

        lengths = [len(obj) for obj in sequences_vec]
        sorted_idx = np.argsort(lengths)

        sequences_vec = self.sort_list_by_idx(sequences_vec, sorted_idx)
        sequences = self.sort_list_by_idx(sequences, sorted_idx)
        labels = self.sort_list_by_idx(labels, sorted_idx)


        meta_batches = []

        i = 0
        while i < bucket_size:
            inr = batch_size
            if i + inr > bucket_size:
                inr = bucket_size - i

            max_len = max([len(obj) for obj in sequences_vec[i:i + inr]])
            inr_ = inr

            j = copy.deepcopy(i)
            batches = []
            while j < i + inr:
                sequences_vec_, input_masks = self.pad(sequences_vec[j:j+inr_], PAD=self.PAD)

                batch = {}
                batch["sequences_vec"] = T.tensor(sequences_vec_).long()
                batch["sequences"] = sequences[j:j+inr_]
                if "listops" in self.config["dataset"] and self.config["parse_trees"]:
                    batch["parse_trees"] = T.tensor(self.parse_trees(sequences[j:j + inr_])).float()
                batch["labels"] = T.tensor(labels[j:j+inr_]).long()
                batch["input_masks"] = T.tensor(input_masks).float()
                batch["batch_size"] = inr_
                batches.append(batch)
                j += inr_
            i += inr

            meta_batches.append(batches)

        random.shuffle(meta_batches)

        batches = []
        for batch_list in meta_batches:
            batches = batches + batch_list

        return batches
