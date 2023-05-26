import pickle
import random
from os import fspath
from pathlib import Path
import numpy as np
from preprocess_tools.process_utils import jsonl_save
import csv
import copy
import statistics

SEED = 101
dev_keys = ["normal"]
test_keys = ["dg8", "100dg8", "LRA", "iid_arg", "ood_10arg", "ood_15arg",
             "200_300", "300_400", "400_500", "500_600", "600_700", "700_800", "800_900", "900_1000"]
np.random.seed(SEED)
random.seed(SEED)
max_seq_len = 100

train_path = Path('../data/listops/train_ndr50.tsv')
dev_path = {}
dev_path["normal"] = Path('../data/listops/dev_ndr50.tsv')
test_path = {}
test_path["dg8"] = Path('../data/listops/test_ndr50.tsv')
test_path["100dg8"] = Path('../data/listops/test_ndr100.tsv')
test_path["LRA"] = Path('../data/listops/basic_test.tsv')


for key in test_keys:
    if key != "dg8" and key != "LRA" and key != "100dg8":
        test_path[key] = Path('../data/listops/test_{}.tsv'.format(key))

Path('../processed_data/listops_ndr50/').mkdir(parents=True, exist_ok=True)

train_save_path = Path('../processed_data/listops_ndr50/train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('../processed_data/listops_ndr50/dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('../processed_data/listops_ndr50/test_{}.jsonl'.format(key))

metadata_save_path = fspath(Path("../processed_data/listops_ndr50/metadata.pkl"))

labels2idx = {}
vocab2count = {}


def updateVocab(word, x=False):
    global vocab2count
    if x:
        if word not in vocab2count:
            raise ValueError(word)
    vocab2count[word] = vocab2count.get(word, 0) + 1


def verify(seq, label):
    global labels2idx
    revlabels2idx = {v: k for k, v in labels2idx.items()}
    label = revlabels2idx[label]
    op_list = ["[MED", "[SM", "[MIN", "[MAX"]
    while len(seq) > 1:
        new_seq = []
        curr_list = []
        for item in seq:
            if item in op_list:
                if curr_list:
                    new_seq += curr_list
                    curr_list = [item]
                else:
                    curr_list.append(item)
            elif item == "]":
                if curr_list:
                    op = curr_list[0]
                    # print(curr_list[1:])
                    items = [int(x) for x in curr_list[1:]]
                    if op == "[MIN":
                        val = min(items)
                    elif op == "[MAX":
                        val = max(items)
                    elif op == "[SM":
                        val = sum(items) % 10
                    elif op == "[MED":
                        val = statistics.median(items)
                    val = str(int(val))
                    new_seq.append(val)
                    curr_list = []
                else:
                    new_seq.append(item)
            else:
                if curr_list:
                    curr_list.append(item)
                else:
                    new_seq.append(item)
        seq = new_seq

    ans = seq[0]
    assert ans == label


def process_data(filename, update_vocab=True, skip_first_row=False, reverse=False):
    global labels2idx

    print("\n\nOpening directory: {}\n\n".format(filename))

    sequences = []
    labels = []
    count = 0
    with open(filename) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for i, row in enumerate(rd):
            if skip_first_row and i == 0:
                continue

            if reverse:
                label = row[1].strip()
                sequence = row[0].strip()
            else:
                label = row[0].strip()
                sequence = row[1].strip()

            if sequence == "":
                continue
            sequence = sequence.replace("( ", "").replace(" )", "").split(" ")

            if label not in labels2idx:
                labels2idx[label] = len(labels2idx)
            label_id = labels2idx[label]

            sequences.append(sequence)
            labels.append(label_id)

            if update_vocab:
                for word in sequence:
                    updateVocab(word, x=skip_first_row)

            count += 1

            if count % 1000 == 0:
                print("Processing Data # {}...".format(count))

    return sequences, labels


train_sequences, train_labels = process_data(train_path)
dev_sequences = {}
dev_labels = {}
for key in dev_keys:
    dev_sequences[key], dev_labels[key] = process_data(dev_path[key])


test_sequences = {}
test_labels = {}

for key in test_keys:
    skip_first_row = False
    reverse = False
    if key == "LRA":
        skip_first_row = True
        reverse = True
    test_sequences[key], test_labels[key] = process_data(test_path[key],
                                                         update_vocab=False,
                                                         skip_first_row=skip_first_row,
                                                         reverse=reverse)


for key in test_keys:
    len_dists = {}

    j = 0
    for seq, label in zip(test_sequences[key], test_labels[key]):
        verify(seq, label)
        seqlen = len(seq)
        #depth = depth_compute(" ".join(seq)) #.count('[')
        #assert depth > 0
        len_dists[seqlen] = len_dists.get(seqlen, 0) + 1
        j += 1


    print("test key: ", key)
    print("\n")
    print("len_dists: ", len_dists)

vocab = [char for char in vocab2count]
vocab += ["<UNK>", "<PAD>", "<SEP>"]

print("train len: ", len(train_sequences))
print("dev len: ", len(dev_sequences["normal"]))
for key in test_keys:
    print("test len (key: {}): ".format(key), len(test_sequences[key]))

vocab2idx = {word: id for id, word in enumerate(vocab)}


def text_vectorize(text):
    return [vocab2idx[word] for word in text]


def vectorize_data(sequences, labels):
    data_dict = {}
    sequences_vec = [text_vectorize(sequence) for sequence in sequences]
    data_dict["sequence"] = sequences
    data_dict["sequence_vec"] = sequences_vec
    data_dict["label"] = labels
    return data_dict


train_data = vectorize_data(train_sequences, train_labels)
dev_data = {}
for key in dev_keys:
    dev_data[key] = vectorize_data(dev_sequences[key], dev_labels[key])
test_data = {}
for key in test_keys:
    test_data[key] = vectorize_data(test_sequences[key], test_labels[key])

jsonl_save(filepath=train_save_path,
           data_dict=train_data)

for key in dev_keys:
    jsonl_save(filepath=dev_save_path[key],
               data_dict=dev_data[key])

for key in test_keys:
    jsonl_save(filepath=test_save_path[key],
               data_dict=test_data[key])

metadata = {"labels2idx": labels2idx,
            "vocab2idx": vocab2idx,
            "dev_keys": dev_keys,
            "test_keys": test_keys}

with open(metadata_save_path, 'wb') as outfile:
    pickle.dump(metadata, outfile)
