# Simple TF IDF Example with no dependencies

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
| .seacrh(query) | return similarity score of each document to query |

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

## Thank You
