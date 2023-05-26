import math
import pickle
import random
from os import fspath
from pathlib import Path
import csv
import jsonlines
import nltk
import numpy as np

from preprocess.preprocess_tools.process_utils import load_glove, jsonl_save

SEED = 101
MAX_VOCAB = 50000
MIN_FREQ = 3
WORDVECDIM = 300
max_seq_len = 150

dev_keys = ["normal"]
test_keys = ["matched", "mismatched",
             "antonym_matched", "antonym_mismatched",
             "length_matched", "length_mismatched",
             "negation_matched", "negation_mismatched",
             "numerical",
             "content_word_swap_matched", "content_word_swap_mismatched",
             "function_word_swap_matched", "function_word_swap_mismatched",
             "keyboard_swap_matched", "keyboard_swap_mismatched",
             "swap_matched", "swap_mismatched",
             "word_overlap_matched", "word_overlap_mismatched"]
np.random.seed(SEED)
random.seed(SEED)

train_path = Path('../data/MNLI/multinli_1.0_train.jsonl')
test_path = {}
test_path["matched"] = Path('../data/MNLI/multinli_1.0_dev_matched.jsonl')
test_path["mismatched"] = Path('../data/MNLI/multinli_1.0_dev_mismatched.jsonl')
test_path["antonym_matched"] = Path('../data/MNLI/Antonym/multinli_0.9_antonym_matched.jsonl')
test_path["antonym_mismatched"] = Path('../data/MNLI/Antonym/multinli_0.9_antonym_mismatched.jsonl')
test_path["length_matched"] = Path('../data/MNLI/Length_Mismatch/multinli_0.9_length_mismatch_matched.jsonl')
test_path["length_mismatched"] = Path('../data/MNLI/Length_Mismatch/multinli_0.9_length_mismatch_mismatched.jsonl')
test_path["negation_matched"] = Path('../data/MNLI/Negation/multinli_0.9_negation_matched.jsonl')
test_path["negation_mismatched"] = Path('../data/MNLI/Negation/multinli_0.9_negation_mismatched.jsonl')
test_path["numerical"] = Path('../data/MNLI/Numerical_Reasoning/multinli_0.9_quant_hard.jsonl')
test_path["content_word_swap_matched"] = Path('../data/MNLI/Spelling_Error/multinli_0.9_dev_gram_contentword_swap_perturbed_matched.jsonl')
test_path["content_word_swap_mismatched"] = Path('../data/MNLI/Spelling_Error/multinli_0.9_dev_gram_contentword_swap_perturbed_mismatched.jsonl')
test_path["function_word_swap_matched"] = Path('../data/MNLI/Spelling_Error/multinli_0.9_dev_gram_functionword_swap_perturbed_matched.jsonl')
test_path["function_word_swap_mismatched"] = Path('../data/MNLI/Spelling_Error/multinli_0.9_dev_gram_functionword_swap_perturbed_mismatched.jsonl')
test_path["keyboard_swap_matched"] = Path('../data/MNLI/Spelling_Error/multinli_0.9_dev_gram_keyboard_matched.jsonl')
test_path["keyboard_swap_mismatched"] = Path('../data/MNLI/Spelling_Error/multinli_0.9_dev_gram_keyboard_mismatched.jsonl')
test_path["swap_matched"] = Path('../data/MNLI/Spelling_Error/multinli_0.9_dev_gram_swap_matched.jsonl')
test_path["swap_mismatched"] = Path('../data/MNLI/Spelling_Error/multinli_0.9_dev_gram_swap_mismatched.jsonl')
test_path["word_overlap_matched"] = Path('../data/MNLI/Word_Overlap/multinli_0.9_taut2_matched.jsonl')
test_path["word_overlap_mismatched"] = Path('../data/MNLI/Word_Overlap/multinli_0.9_taut2_mismatched.jsonl')


embedding_path = Path("../embeddings/glove/glove.840B.300d.txt")

Path('../processed_data/MNLIdev/').mkdir(parents=True, exist_ok=True)


train_save_path = Path('../processed_data/MNLIdev/train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('../processed_data/MNLIdev/dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('../processed_data/MNLIdev/test_{}.jsonl'.format(key))
metadata_save_path = fspath(Path("../processed_data/MNLIdev/metadata.pkl"))

labels2idx = {}
vocab2count = {}


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def updateVocab(word):
    global vocab2count
    vocab2count[word] = vocab2count.get(word, 0)+1


def process_data(filename, update_vocab=True, filter=False):

    global labels2idx

    print("\n\nOpening directory: {}\n\n".format(filename))

    sequences1 = []
    sequences2 = []
    labels = []
    count = 0

    if "jsonl" in fspath(filename):
        with jsonlines.open(filename) as reader:
            for sample in reader:
                if sample['gold_label'] != '-':

                    sequence1 = tokenize(sample['sentence1'].lower())
                    sequence2 = tokenize(sample['sentence2'].lower())
                    label = sample['gold_label']
                    if label not in labels2idx:
                        labels2idx[label] = len(labels2idx)
                    label_id = labels2idx[label]

                    if filter:
                        if (len(sequence1) < max_seq_len) and (len(sequence2) < max_seq_len):
                            sequences1.append(sequence1)
                            sequences2.append(sequence2)
                            labels.append(label_id)
                    else:
                        sequences1.append(sequence1)
                        sequences2.append(sequence2)
                        labels.append(label_id)

                    if update_vocab:
                        for word in sequence1:
                            updateVocab(word)

                        for word in sequence2:
                            updateVocab(word)

                    count += 1

                    if count % 1000 == 0:
                        print("Processing Data # {}...".format(count))
    elif "tsv" in fspath(filename):
        with open(filename) as tsv_file:
            read_tsv = csv.reader(tsv_file, delimiter="\t")
            for i, row in enumerate(read_tsv):
                if i > 0:
                    if row[2] != '-':
                        sequence1 = tokenize(row[0].lower())
                        sequence2 = tokenize(row[1].lower())
                        label = row[2]
                        if label not in labels2idx:
                            labels2idx[label] = len(labels2idx)
                        label_id = labels2idx[label]

                        if filter:
                            if (len(sequence1) < max_seq_len) and (len(sequence2) < max_seq_len):
                                sequences1.append(sequence1)
                                sequences2.append(sequence2)
                                labels.append(label_id)
                        else:
                            sequences1.append(sequence1)
                            sequences2.append(sequence2)
                            labels.append(label_id)

                        if update_vocab:
                            for word in sequence1:
                                updateVocab(word)

                            for word in sequence2:
                                updateVocab(word)

                        count += 1

                        if count % 1000 == 0:
                            print("Processing Data # {}...".format(count))


    return sequences1, sequences2, labels


train_sequences1, \
    train_sequences2, \
    train_labels = process_data(train_path, filter=True)

train_idx = [id for id in range(len(train_sequences1))]
random.shuffle(train_idx)

def sort(objs, idx):
    return [objs[id] for id in idx]

train_sequences1 = sort(train_sequences1, train_idx)
train_sequences2 = sort(train_sequences2, train_idx)
train_labels = sort(train_labels, train_idx)

dev_sequences1 = {}
dev_sequences2 = {}
dev_labels = {}

for key in dev_keys:
    dev_sequences1[key] = train_sequences1[0:10000]
    dev_sequences2[key] = train_sequences2[0:10000]
    dev_labels[key] = train_labels[0:10000]

train_sequences1 = train_sequences1[10000:]
train_sequences2 = train_sequences2[10000:]
train_labels = train_labels[10000:]

test_sequences1 = {}
test_sequences2 = {}
test_labels = {}

for key in test_keys:
    test_sequences1[key], \
        test_sequences2[key], \
        test_labels[key] = process_data(test_path[key], update_vocab=True)

counts = []
vocab = []
for word, count in vocab2count.items():
    if count > MIN_FREQ:
        vocab.append(word)
        counts.append(count)

vocab2embed = load_glove(embedding_path, vocab=vocab2count, dim=WORDVECDIM)

sorted_idx = np.flip(np.argsort(counts), axis=0)
vocab = [vocab[id] for id in sorted_idx if vocab[id] in vocab2embed]
if len(vocab) > MAX_VOCAB:
    vocab = vocab[0:MAX_VOCAB]


vocab += ["<PAD>", "<UNK>", "<SEP>"]

print(vocab)

vocab2idx = {word: id for id, word in enumerate(vocab)}

vocab2embed["<PAD>"] = np.zeros((WORDVECDIM), np.float32)
b = math.sqrt(3/WORDVECDIM)
vocab2embed["<UNK>"] = np.random.uniform(-b, +b, WORDVECDIM)
vocab2embed["<SEP>"] = np.random.uniform(-b, +b, WORDVECDIM)

embeddings = []
for id, word in enumerate(vocab):
    embeddings.append(vocab2embed[word])


def text_vectorize(text):
    return [vocab2idx.get(word, vocab2idx['<UNK>']) for word in text]


def vectorize_data(sequences1, sequences2, labels):
    data_dict = {}
    sequences1_vec = [text_vectorize(sequence) for sequence in sequences1]
    sequences2_vec = [text_vectorize(sequence) for sequence in sequences2]
    data_dict["sequence1"] = sequences1
    data_dict["sequence2"] = sequences2
    sequences_vec = [sequence1 + [vocab2idx["<SEP>"]] + sequence2 for sequence1, sequence2 in
                     zip(sequences1_vec, sequences2_vec)]
    data_dict["sequence1_vec"] = sequences1_vec
    data_dict["sequence2_vec"] = sequences2_vec
    data_dict["sequence_vec"] = sequences_vec
    data_dict["label"] = labels
    return data_dict


train_data = vectorize_data(train_sequences1, train_sequences2, train_labels)
"""
for item in train_data["sequence1"]:
    print(item)
print("\n\n")
"""
dev_data = {}
for key in dev_keys:
    dev_data[key] = vectorize_data(dev_sequences1[key], dev_sequences2[key], dev_labels[key])
test_data = {}
for key in test_keys:
    test_data[key] = vectorize_data(test_sequences1[key], test_sequences2[key], test_labels[key])

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