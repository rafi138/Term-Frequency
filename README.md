# Simple TF IDF Example with no dependencies

Requires Python 3.13 (see Pipfile). The `tfidf.py` module itself only uses the standard library.

### How to use !
  - Download or clone this repo https://github.com/rafi138/Term-Frequency (using terminal or command prompt)
```sh
$ git clone https://github.com/rafi138/Term-Frequency
```
  - import tfidf.py & make object of it
```sh
from tfidf import tfIDF
# dataset (make sure it is a list)
dataset = ['my name is tfidf', 'her name is feature extraction']

# make an object
tfidf_object = tfIDF(dataset)
```
Make sure you have python installed

there are several function you can call to get some info from tfidf object

| Function | Descriptions |
| ------ | ------ |
| .getTFIDF() | return tfidf score of each word |
| .getSCORE() | return tfidf score of each document |
| .getIDF() | return idf score of each word |
| .getWORDSET() | return all word in dataset (all document) |
| .getWORDDICT() | return term frequency of each word |
| .search(query) | return similarity score of each document to query (matches whole words, case-insensitive) |

The constructor validates its input: the dataset must be a non-empty list (or tuple) of strings, and `search()` requires a string query, otherwise a `TypeError` or `ValueError` is raised.

if want to simply learn how to use it or see the result, just run my jupyter notebook file (TFIDF.ipynb)

### Example

```sh
from tfidf import tfIDF
# dataset (make sure it is a list)
dataset = ['my name is tfidf', 'her name is feature extraction']

# make an object
tfidf_object = tfIDF(dataset)

# searching

print(tfidf_object.search('tfidf'))
# it will print index & similarity score of document
# OUTPUT : [[0, 0.1812381165789131], [1, 0.0]]
```

### Feature extraction script

`feature_extract.py` reads any text file given as a command line argument and prints trigram features plus a TF-IDF ranking:

```sh
pipenv install
pipenv run python feature_extract.py path/to/your/textfile.txt
```

On first run it downloads the nltk resources it needs (`stopwords`, `punkt`). The test suite runs without any dependencies:

```sh
python3 test.py
```

## Thank You
