import copy
import math
import pickle
import random
from os import fspath
from pathlib import Path
import numpy as np
from preprocess_tools.process_utils import load_glove, jsonl_save
import csv
import os
import nltk

embedding_path = Path("../embeddings/glove/glove.840B.300d.txt")
MAX_VOCAB = 50000
MIN_FREQ = 5
WORDVECDIM = 300

SEED = 101
dev_keys = ["normal"]
test_keys = ["normal",
             "original_of_contrast",
             "contrast",
             "original_of_counterfactual",
             "counterfactual"]
np.random.seed(SEED)
random.seed(SEED)

train_path = Path('../data/IMDB/aclImdb/train')
dev_path1 = Path('../data/IMDB/dev_original.tsv')
dev_path2 = Path('../data/IMDB/dev_contrast.tsv')
test_path = {}
test_path["normal"] = Path('../data/IMDB/aclImdb/test')
test_path["original_of_contrast"] = Path('../data/IMDB/test_original_contrast.tsv')
test_path["contrast"] = Path('../data/IMDB/test_contrast.tsv')
test_path["original_of_counterfactual"] = Path('../data/IMDB/test_original_counterfactual.tsv')
test_path["counterfactual"] = Path('../data/IMDB/test_counterfactual.tsv')

Path('../processed_data/IMDB').mkdir(parents=True, exist_ok=True)
train_save_path = Path('../processed_data/IMDB/train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('../processed_data/IMDB/dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('../processed_data/IMDB/test_{}.jsonl'.format(key))
test_save_path["contrast_pair"] = Path('../processed_data/IMDB/test_{}.jsonl'.format("contrast_pair"))
metadata_save_path = fspath(Path("../processed_data/IMDB/metadata.pkl"))

labels2idx = {}

vocab2count = {}


def tokenize(sequence):
    return sequence.split()


def updateVocab(word):
    global vocab2count
    vocab2count[word] = vocab2count.get(word, 0) + 1


def process_data1(filename, update_vocab=True):
    global labels2idx

    sequences = []
    labels = []
    sequences2 = []
    labels2 = []
    count = 0
    i = 0

    for root, dirs, files in os.walk(filename, topdown=False):
        for name in files:
            subfilename = os.path.join(root, name)
            if "pos" in subfilename:
                label = "positive"
                with open(subfilename, encoding="utf8") as fp:
                    lines = fp.readlines()
                    sequence = lines[0].lower().replace("<br>", "").replace("</br>", "").replace("<br />", "").strip()
            elif "neg" in subfilename:
                label = "negative"
                with open(subfilename, encoding="utf8") as fp:
                    lines = fp.readlines()
                    sequence = lines[0].lower().replace("<br>", "").replace("</br>", "").replace("<br />", "").strip()
            else:
                continue

            sequence = nltk.word_tokenize(sequence)



            if len(sequence) <= 200 or not update_vocab:
                # print("label {}:".format(i), label)
                i += 1

                if label not in labels2idx:
                    labels2idx[label] = len(labels2idx)
                label_id = labels2idx[label]

                if update_vocab:
                    for word in sequence:
                        updateVocab(word)

                sequences.append(sequence)
                labels.append(label_id)

                count += 1

                if count % 1000 == 0:
                    print("Processing Data # {}...".format(count))
            elif len(sequence) <= 300:
                if label not in labels2idx:
                    labels2idx[label] = len(labels2idx)
                label_id = labels2idx[label]

                sequences2.append(sequence)
                labels2.append(label_id)

    return sequences, labels, sequences2, labels2


train_sequences, train_labels, dev_sequences0, dev_labels0 = process_data1(train_path)

test_sequences = {}
test_labels = {}
test_sequences["normal"], test_labels["normal"], _, _ = process_data1(test_path["normal"])


def process_data2(filename, update_vocab=True):
    global labels2idx

    sequences = []
    labels = []
    count = 0

    with open(filename, encoding="utf8") as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for i, row in enumerate(read_tsv):
            if i == 0:
                continue
            label = row[0].lower()
            sequence = row[1].lower().replace("<br>", "").replace("</br>", "").replace("<br />", "").strip()

            # print("sequence {}:".format(i), sequence)
            # print("label {}:".format(i), label)
            if label not in labels2idx:
                labels2idx[label] = len(labels2idx)
            label_id = labels2idx[label]

            sequence = nltk.word_tokenize(sequence)

            sequences.append(sequence)
            labels.append(label_id)
            if update_vocab:
                for word in sequence:
                    updateVocab(word)

            count += 1

            if count % 1000 == 0:
                print("Processing Data # {}...".format(count))

    return sequences, labels


dev_sequences = {}
dev_labels = {}
dev_sequences1, dev_labels1 = process_data2(dev_path1, update_vocab=False)
dev_sequences2, dev_labels2 = process_data2(dev_path1, update_vocab=False)
dev_sequences["normal"] = dev_sequences0 + dev_sequences1 + dev_sequences2
dev_labels["normal"] = dev_labels0 + dev_labels1 + dev_labels2

for key in test_keys:
    if key != "normal" and "adversarial" not in key:
        test_sequences[key], test_labels[key] = process_data2(test_path[key], update_vocab=False)

print("training size: ", len(train_sequences))
print("development size: ", len(dev_sequences["normal"]))
for key in test_keys:
    print("Test size (key:{})".format(key), len(test_sequences[key]))

print("initial_vocab: ", len(vocab2count))
counts = []
vocab = []
for word, count in vocab2count.items():
    if count >= MIN_FREQ:
        vocab.append(word)
        counts.append(count)
print("filtered vocab: ", len(vocab))

vocab2embed = load_glove(embedding_path, vocab=vocab2count, dim=WORDVECDIM)

print("new vocab: ", len(vocab2embed))

sorted_idx = np.flip(np.argsort(counts), axis=0)
vocab = [vocab[id] for id in sorted_idx if vocab[id] in vocab2embed]
if len(vocab) > MAX_VOCAB:
    vocab = vocab[0:MAX_VOCAB]

vocab += ["<PAD>", "<UNK>", "<SEP>"]

# print(vocab)

vocab2idx = {word: id for id, word in enumerate(vocab)}

vocab2embed["<PAD>"] = np.zeros((WORDVECDIM), np.float32)
b = math.sqrt(3 / WORDVECDIM)
vocab2embed["<UNK>"] = np.random.uniform(-b, +b, WORDVECDIM)
vocab2embed["<SEP>"] = np.random.uniform(-b, +b, WORDVECDIM)

embeddings = []
for id, word in enumerate(vocab):
    embeddings.append(vocab2embed[word])


def text_vectorize(text):
    return [vocab2idx.get(word, vocab2idx['<UNK>']) for word in text]


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
            "embeddings": np.asarray(embeddings, np.float32),
            "dev_keys": dev_keys,
            "test_keys": test_keys}

with open(metadata_save_path, 'wb') as outfile:
    pickle.dump(metadata, outfile)
