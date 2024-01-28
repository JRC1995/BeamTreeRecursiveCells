## Official Code for Beam Tree Recursive Cells (ICML 2023)

[ArXiv Link](https://arxiv.org/abs/2305.19999)

For an extended repo see: https://github.com/JRC1995/BeamRecursionFamily/tree/main

### Credits:
* ```framework, models/layers/geometric, models/layers/ndr_geometric_stack.py``` adapted from: https://github.com/RobertCsordas/ndr
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
* tensorflow-datasets==4.5.2

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

You can verify if the data is properly set up from the directory tree [here](https://github.com/JRC1995/BeamTreeRecursiveCells/blob/main/data_directory_tree.md).

### Processing Data
* Go to preprocess/ and run each preprocess files to preprocess the corresponding data (process_SNLI_addon.py must be run after process_SNLI.py; otherwise no order requirement)

We share some of the processed data with its exact splits [here](https://drive.google.com/file/d/1RMZSI3lOTC7GoVbl5D7QeK0qOgGNI46X/view?usp=sharing) (put the processed_data folder in the outermost project directory).

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



