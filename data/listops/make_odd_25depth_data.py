import random
import numpy as np
import csv

MIN = "[MIN"
MAX = "[MAX"
MED = "[MED"
FIRST = "[FIRST"
LAST = "[LAST"
SUM_MOD = "[SM"
END = "]"

OPERATORS = [MIN, MAX, MED, SUM_MOD]  # , FIRST, LAST]
VALUES = range(10)

VALUE_P = 0.8
MAX_ARGS = 5
MAX_DEPTH = 25

min_seq_len = 100
max_seq_len = 1000
DATA_POINTS = 2000


def generate_tree(depth):
    if depth < MAX_DEPTH:
        r = random.random()
    else:
        r = 1

    if r > VALUE_P:
        value = random.choice(VALUES)
        return value
    else:
        num_values = random.randint(2, MAX_ARGS)
        values = []
        for _ in range(num_values):
            values.append(generate_tree(depth + 1))

        op = random.choice(OPERATORS)
        t = (op, values[0])
        for value in values[1:]:
            t = (t, value)
        t = (t, END)
    return t


def to_string(t, parens=True):
    if isinstance(t, str):
        return t
    elif isinstance(t, int):
        return str(t)
    else:
        if parens:
            return '( ' + to_string(t[0]) + ' ' + to_string(t[1]) + ' )'


def to_value(t):
    if not isinstance(t, tuple):
        return t
    l = to_value(t[0])
    r = to_value(t[1])
    if l in OPERATORS:  # Create an unsaturated function.
        return (l, [r])
    elif r == END:  # l must be an unsaturated function.
        if l[0] == MIN:
            return min(l[1])
        elif l[0] == MAX:
            return max(l[1])
        elif l[0] == FIRST:
            return l[1][0]
        elif l[0] == LAST:
            return l[1][-1]
        elif l[0] == MED:
            return int(np.median(l[1]))
        elif l[0] == SUM_MOD:
            return (np.sum(l[1]) % 10)
    elif isinstance(l, tuple):  # We've hit an unsaturated function and an argument.
        return (l[0], l[1] + [r])

def depth_compute(seq):
    curr_depth = 0
    max_depth = 0
    for c in seq:
        if c == "[":
            curr_depth += 1
            max_depth = max(curr_depth, max_depth)
        elif c == "]":
            curr_depth -= 1

    return max_depth

data = set()
while len(data) < DATA_POINTS:
    example = generate_tree(1)
    tokenized_example = to_string(example).replace("( ", "").replace(" )", "").split(" ")
    print("depth: ", depth_compute(tokenized_example))
    if depth_compute(" ".join(tokenized_example)) > 20:
        data.add(example)
        print(" ".join(tokenized_example))

with open('test_ood_25depth.tsv', 'wt', newline="") as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for example in data:
        tsv_writer.writerow([str(to_value(example)), to_string(example)])
