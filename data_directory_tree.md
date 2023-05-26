```
├───data
│   ├───IMDB
│   │   │   dev_contrast.tsv
│   │   │   dev_original.tsv
│   │   │   test_contrast.tsv
│   │   │   test_counterfactual.tsv
│   │   │
│   │   └───aclImdb
│   │       │   imdb.vocab
│   │       │   imdbEr.txt
│   │       │   README
│   │       │
│   │       ├───test
│   │       │   │   labeledBow.feat
│   │       │   │   urls_neg.txt
│   │       │   │   urls_pos.txt
│   │       │   │
│   │       │   ├───neg
│   │       │   └───pos
│   │       └───train
│   │           │   labeledBow.feat
│   │           │   unsupBow.feat
│   │           │   urls_neg.txt
│   │           │   urls_pos.txt
│   │           │   urls_unsup.txt
│   │           │
│   │           ├───neg
│   │           ├───pos
│   │           └───unsup
│   ├───listops
│   │       base.py
│   │       basic_test.tsv
│   │       dev_d7s.tsv
│   │       load_listops_data.py
│   │       make_depth_dev_data.py
│   │       make_depth_ndr_data.py
│   │       make_depth_test_data.py
│   │       make_depth_train_data.py
│   │       make_depth_train_data_extra.py
│   │       make_iid_data.py
│   │       make_odd_25depth_data.py
│   │       make_ood_10arg_data.py
│   │       make_ood_15arg_data.py
│   │       test_200_300.tsv
│   │       test_300_400.tsv
│   │       test_400_500.tsv
│   │       test_500_600.tsv
│   │       test_600_700.tsv
│   │       test_700_800.tsv
│   │       test_800_900.tsv
│   │       test_900_1000.tsv
│   │       test_d20s.tsv
│   │       test_dg8s.tsv
│   │       test_iid_arg.tsv
│   │       test_ood_10arg.tsv
│   │       test_ood_15arg.tsv
│   │       train_d20s.tsv
│   │       train_d6s.tsv
│   │       __init__.py
│   │
│   ├───listops_lra
│   │       basic_test.tsv
│   │       basic_train.tsv
│   │       basic_val.tsv
│   │
│   ├───MNLI
│   │   │   conj_dev.tsv
│   │   │   multinli_0.9_test_matched_unlabeled.jsonl
│   │   │   multinli_0.9_test_matched_unlabeled_hard.jsonl
│   │   │   multinli_0.9_test_mismatched_unlabeled.jsonl
│   │   │   multinli_0.9_test_mismatched_unlabeled.jsonl.zip
│   │   │   multinli_0.9_test_mismatched_unlabeled_hard.jsonl
│   │   │   multinli_1.0_dev_matched.jsonl
│   │   │   multinli_1.0_dev_matched.txt
│   │   │   multinli_1.0_dev_mismatched.jsonl
│   │   │   multinli_1.0_dev_mismatched.txt
│   │   │   multinli_1.0_train.jsonl
│   │   │   multinli_1.0_train.txt
│   │   │   paper.pdf
│   │   │   README.txt
│   │   │
│   │   ├───Antonym
│   │   │       multinli_0.9_antonym_matched.jsonl
│   │   │       multinli_0.9_antonym_matched.txt
│   │   │       multinli_0.9_antonym_mismatched.jsonl
│   │   │       multinli_0.9_antonym_mismatched.txt
│   │   │
│   │   ├───Length_Mismatch
│   │   │       multinli_0.9_length_mismatch_matched.jsonl
│   │   │       multinli_0.9_length_mismatch_matched.txt
│   │   │       multinli_0.9_length_mismatch_mismatched.jsonl
│   │   │       multinli_0.9_length_mismatch_mismatched.txt
│   │   │
│   │   ├───Negation
│   │   │       multinli_0.9_negation_matched.jsonl
│   │   │       multinli_0.9_negation_matched.txt
│   │   │       multinli_0.9_negation_mismatched.jsonl
│   │   │       multinli_0.9_negation_mismatched.txt
│   │   │
│   │   ├───Numerical_Reasoning
│   │   │       .DS_Store
│   │   │       multinli_0.9_quant_hard.jsonl
│   │   │       multinli_0.9_quant_hard.txt
│   │   │
│   │   ├───Spelling_Error
│   │   │       multinli_0.9_dev_gram_contentword_swap_perturbed_matched.jsonl
│   │   │       multinli_0.9_dev_gram_contentword_swap_perturbed_matched.txt
│   │   │       multinli_0.9_dev_gram_contentword_swap_perturbed_mismatched.jsonl
│   │   │       multinli_0.9_dev_gram_contentword_swap_perturbed_mismatched.txt
│   │   │       multinli_0.9_dev_gram_functionword_swap_perturbed_matched.jsonl
│   │   │       multinli_0.9_dev_gram_functionword_swap_perturbed_matched.txt
│   │   │       multinli_0.9_dev_gram_functionword_swap_perturbed_mismatched.jsonl
│   │   │       multinli_0.9_dev_gram_functionword_swap_perturbed_mismatched.txt
│   │   │       multinli_0.9_dev_gram_keyboard_matched.jsonl
│   │   │       multinli_0.9_dev_gram_keyboard_matched.txt
│   │   │       multinli_0.9_dev_gram_keyboard_mismatched.jsonl
│   │   │       multinli_0.9_dev_gram_keyboard_mismatched.txt
│   │   │       multinli_0.9_dev_gram_swap_matched.jsonl
│   │   │       multinli_0.9_dev_gram_swap_matched.txt
│   │   │       multinli_0.9_dev_gram_swap_mismatched.jsonl
│   │   │       multinli_0.9_dev_gram_swap_mismatched.txt
│   │   │
│   │   └───Word_Overlap
│   │           multinli_0.9_taut2_matched.jsonl
│   │           multinli_0.9_taut2_matched.txt
│   │           multinli_0.9_taut2_mismatched.jsonl
│   │           multinli_0.9_taut2_mismatched.txt
│   │
│   ├───proplogic
│   │       generate_neg_set_data.py
│   │       test0
│   │       test1
│   │       test10
│   │       test11
│   │       test12
│   │       test2
│   │       test3
│   │       test4
│   │       test5
│   │       test6
│   │       test7
│   │       test8
│   │       test9
│   │       train0
│   │       train1
│   │       train10
│   │       train11
│   │       train12
│   │       train2
│   │       train3
│   │       train4
│   │       train5
│   │       train6
│   │       train7
│   │       train8
│   │       train9
│   │       __init__.py
├───embeddings
│   └───glove
│           glove.840B.300d.txt
```
