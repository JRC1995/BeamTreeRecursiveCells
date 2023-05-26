## Official Code for Beam Tree Recursive Cells (ICML 2023)

### Credits:
* ```framework, models/layers/geometric, models/layers/ndr_geometric_stack.py``` adapted from: https://github.com/RobertCsordas/ndr
* CRvNN-related models are adapted from: https://github.com/JRC1995/Continuous-RvNN
* ```models/encoder/OrderedMemory.py``` adapted from: https://github.com/yikangshen/Ordered-Memory
* Gumbel-Tree, BT-GRC, and GT-GRC related models are adapted from: https://github.com/jihunchoi/unsupervised-treelstm
* ```optimizers/ranger.py``` adapted from: https://github.com/anoidgit/transformer/blob/master/optm/ranger.py
* models/topk_module.py taken from the supplementary code in: https://proceedings.neurips.cc/paper/2020/hash/ec24a54d62ce57ba93a531b460fa8d18-Abstract.html

### Requirements
* torch==1.10.0
* tqdm==4.62.3
* jsonlines==2.0.0
* torchtext==0.8.1
* ninja==1.10.2
* typing-extensions==4.5.0
* psutil==5.8.0

### Data Setup
* Put the Logical Inference data files (train0,train1,train2,.....test12) (downloaded from https://github.com/yikangshen/Ordered-Memory/tree/master/data/propositionallogic) in data/proplogic/
* Download the ListOps data (along with extrapolation data) from the urls here: https://github.com/facebookresearch/latent-treelstm/blob/master/data/listops/external/urls.txt and put the tsv files in data/listops/
* Run all the make*.py files in data/listops/ to create relevant splits (exact splits used in the paper will be released later) 
* Download LRA (https://github.com/google-research/long-range-arena) dataset
* From LRA dataset put the ListOps basic_test.tsv (LRA test set) in data/listops
* Download IMDB original split from here: http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz, extract and put the acllmdb folder in data/IMDB/ 
* Download IMDB contrast set from here: https://github.com/allenai/contrast-sets/tree/main/IMDb/data, put the dev_original.tsv, dev_contrast.tsv and test_contrast.tsv in data/IMDB/
* Download IMDB counterfactual test set from here: https://github.com/acmi-lab/counterfactually-augmented-data/blob/master/sentiment/new/test.tsv; rename it to "test_counterfactual.tsv". Put it in data/IMDB
* Download MNLI datasets (https://cims.nyu.edu/~sbowman/multinli/) and put them in data/MNLI/
* Download MNLI stress tests (https://abhilasharavichander.github.io/NLI_StressTest/) and put them in data/MNLI/
* Put glove.840B.300d.txt (download glove.840B.300d.zip from here:https://nlp.stanford.edu/projects/glove/) in embeddings/glove/

You can verify if the data is properly set up from the tree below.
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
### Processing Data
* Go to preprocess/ and run each preprocess files to preprocess the corresponding data (process_SNLI_addon.py must be run after process_SNLI.py; otherwise no order requirement)

We share some of the processed data with its exact splits here (put the processed_data folder in the outermost project directory).

### How to train
Train:
```python trian.py --model=[insert model name] -- dataset=[insert dataset name] --times=[insert total runs] --device=[insert device name] --model_type=[classifier/sentence_pair]```

* Check argparser.py for exact options. 
* Model type ```sentence_pair``` represents sentence-matching tasks like NLI. Modely type ```classifier``` represents simple sentence classification tasks.
* Generally we use total times as 3. 


### Tree Parsing
* For tree parsing from a classifier model: ```python extract_trees_classifier.py --model=[insert model name] --device=[insert device name] -- dataset=[insert dataset name]```
* For tree parsing from a sentence-paur matching model: ```python extract_trees_nli.py --model=[insert model name] --device=[insert device name] -- dataset=[insert dataset name]```

Inputs for parsing can be modified from inside a list in ```extract_trees_classifier.py``` or ```python extract_trees_nli.py ``` (line 66)


### Dataset Nomenclature
The dataset nomenclature in the codebase and in the paper are a bit different. We provide a mapping here of the form ([codebase dataset name] == [paper dataset name])

* listopsc == ListOps
* listopsd == ListOps-DG
* listops_ndr50 == ListOps-DG1
* listops_ndr100 == ListOps-DG2
* proplogic == Logical Inference (Operator generalization split)
* proplogic_C == Logical Inference (C-split for systematic generalization)
* SST2 == SST2
* SST5 == SST5
* IMDB == IMDB
* MNLIdev == MNLI

The speed-suffixed names are for stress tests. 

### Model Nomenclature
The model nomenclature in the codebase and in the paper are a bit different. We provide a mapping here of the form ([codebase model name] == [paper model name])

* RCell == RecurrentGRC
* BalancedTreeCell == BalancedTreeGRC
* RandomTreeCell == RandomTreeGRC
* GoldTreeCell == GoldTreeGRC
* GumbelTreeLSTM == GumbelTreeLSTM
* GumbelTreeCell == GumbelTreeGRC
* MCGumbelTreeCell == MCGumbelTreeGRC
* CYKCell == CYK-GRC
* OrderedMemory = Ordered Memory  
* CRvNN == CRvNN
* CRvNN_worst == CRvNN without halt (during stress test)  
* BSRPCell == BSRP-GRC (beam 5)
* BigBSRPCell == BSRP-GRC (beam 8)
* NDR = NDR (Neural Data Router)
* BeamTreeLSTM == BT-LSTM (beam 5)
* BeamTreeCell == BT-GRC (beam 5)
* SmallerBeamTreeCell == BT-GRC (beam 2)
* DiffBeamTreeCell == BT-GRC + OneSoft (beam 5)
* SmallerDiffBeamTreeCell == BT-GRC + OneSoft (beam 2)
* DiffSortBeamTreeCell == BT-GRC + SOFT (beam 5)

### Citation

```
@InProceedings{Chowdhury2023beam,
  title = 	 {Beam Tree Recursive Cells},
  author =       {Ray Chowdhury, Jishnu and Caragea, Cornelia},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  year = 	 {2023}
}
```
Contact the associated github email for any question or issue. 


