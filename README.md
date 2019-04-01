# Code2Comment
This is a part of CS253 Final project. Mainly the code processing (convert code from plain text to structure-embedded text), and the baseline model (naive element-wise teacher-forcing LSTM). 

## Parse
Some raw codes are in python2, or simply cannot be parsed to AST. So check all records if AST feasible, then remove failures. Note python2 code will be converted to python3 and then saved.

**Usage**

`python code_parse.py`

This will check `train.txt`, `valid.txt` and `test.txt` in a `raw_data` dir, and save all feasible records as pickle to a new dir `data`, filename respectively.


## Encode
To encode code using SBT, we need following steps:
* Traverse AST trees of all code in train set (only train for encoding), find all unique user-defined variable (or function, argument, class, attribute, module, string, number) names, select the most frequent $N$ as the 'namespace' vocabulary.

* SBT traverse all code in train set, save all unique tokens as dictionary. This requires a existed 'namespace' vocabulary.

* encode and decode using one-hot or other embedding. Trivial thing.

**Usage**

`python SBT_encode.py`

This will call `build_vocab` and build the vocabulary.

**Classes**

* `Travese`: traverse an AST tree, save all unique arguments (Code attributes) and names (User-defined names)

* `Sbt`: traverse an AST tree, convert to SBT format.

* `Encoder`: Tool to encode a piece of code. Requires two files, `vocab_dict.pkl`, the SBT tokens vocabulary, and `frequently_used_name.pkl`, 'namespace' vocabulary. The latter is required because SBT treats frequent and rare user-defined names differently.

**Functions**

* `build_vocab`: read the cleaned train data `data/train.pkl`, do name traverse then SBT traverse, generate respective files `frequently_used_name.pkl` and `vocab_dict.pkl`.






