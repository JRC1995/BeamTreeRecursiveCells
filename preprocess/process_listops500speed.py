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
test_keys = ["normal"]
np.random.seed(SEED)
random.seed(SEED)

min_seq_len = 0
max_seq_len = 1000000000
max_items = 100

train_path = Path('../data/listops/test_500_600.tsv')
Path('../processed_data/listops500speed/').mkdir(parents=True, exist_ok=True)

train_save_path = Path('../processed_data/listops500speed/train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('../processed_data/listops500speed/dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('../processed_data/listops500speed/test_{}.jsonl'.format(key))

metadata_save_path = fspath(Path("../processed_data/listops500speed/metadata.pkl"))

labels2idx = {}
vocab2count = {}
def updateVocab(word, x=False):
    global vocab2count
    if x:
        if word not in vocab2count:
            raise ValueError(word)
    vocab2count[word] = vocab2count.get(word, 0) + 1


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

            if len(sequence) <= max_seq_len and len(sequence) >= min_seq_len:
                sequences.append(sequence)
                labels.append(label_id)

                if update_vocab:
                    for word in sequence:
                        updateVocab(word, x=skip_first_row)

                count += 1

                if count == 100:
                    print("hello")
                    break
    return sequences, labels


train_sequences, train_labels = process_data(train_path)

dev_sequences = {"normal": train_sequences}
dev_labels = {"normal": train_labels}

test_sequences = {"normal": train_sequences}
test_labels = {"normal": train_labels}

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
