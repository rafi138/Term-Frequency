# Importing libraries
import argparse
import re
import sys

import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract trigram features and TF-IDF ranking from a text file')
    parser.add_argument('input_file',
                        help='path to the text file to analyze')
    return parser.parse_args()


def read_input_file(path):
    # Read the input file, exiting with a clear message instead of a traceback
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as file:
            lines = file.readlines()
    except FileNotFoundError:
        sys.exit('Error: file not found: %s' % path)
    except IsADirectoryError:
        sys.exit('Error: path is a directory, not a file: %s' % path)
    except PermissionError:
        sys.exit('Error: permission denied: %s' % path)
    except OSError as exc:
        sys.exit('Error: could not read file %s: %s' % (path, exc))
    if not lines:
        sys.exit('Error: input file is empty: %s' % path)
    return lines


def ensure_nltk_data():
    # Download the nltk resources used below when they are missing
    try:
        stopwords.words('english')
    except LookupError:
        if not nltk.download('stopwords'):
            sys.exit('Error: could not download nltk resource: stopwords')
    try:
        nltk.word_tokenize('word')
    except LookupError:
        for resource in ('punkt', 'punkt_tab'):
            try:
                downloaded = nltk.download(resource)
            except Exception:
                downloaded = False
            if not downloaded:
                # older nltk versions have no punkt_tab package; punkt suffices
                if resource == 'punkt_tab':
                    continue
                sys.exit('Error: could not download nltk resource: %s' % resource)


# Preprocessing
def remove_string_special_characters(s):

    # removes special characters with ' '
    stripped = re.sub(r'[^a-zA-Z\s]', '', s)

    # Change any white space to one space
    stripped = re.sub(r'\s+', ' ', stripped)

    # Remove start and end white spaces
    stripped = stripped.strip()
    if stripped != '':
            return stripped.lower()
    return ''


def main():
    args = parse_args()
    txt1 = read_input_file(args.input_file)
    ensure_nltk_data()

    # Stopword removal
    stop_words = set(stopwords.words('english'))
    your_list = ['skills', 'ability', 'job', 'description']
    for i, line in enumerate(txt1):
        txt1[i] = ' '.join([x for
            x in nltk.word_tokenize(line) if
            ( x not in stop_words ) and ( x not in your_list )])

    if not any(txt1):
        sys.exit('Error: no usable text left after preprocessing: %s' % args.input_file)

    # Getting trigrams
    vectorizer = CountVectorizer(ngram_range = (3,3))
    try:
        X1 = vectorizer.fit_transform(txt1)
    except ValueError as exc:
        sys.exit('Error: could not extract features from input: %s' % exc)
    features = (vectorizer.get_feature_names_out())
    print("\n\nFeatures : \n", features)
    print("\n\nX1 : \n", X1.toarray())

    # Applying TFIDF
    vectorizer = TfidfVectorizer(ngram_range = (3,3))
    try:
        X2 = vectorizer.fit_transform(txt1)
    except ValueError as exc:
        sys.exit('Error: could not compute TF-IDF on input: %s' % exc)
    scores = (X2.toarray())
    print("\n\nScores : \n", scores)

    # Getting top ranking features
    # Use the feature names of the vectorizer that produced the scores;
    # mixing them with CountVectorizer's vocabulary misaligns terms and ranks
    sums = X2.sum(axis = 0)
    tfidf_features = vectorizer.get_feature_names_out()
    data1 = []
    for col, term in enumerate(tfidf_features):
        data1.append( (term, sums[0,col] ))
    ranking = pd.DataFrame(data1, columns = ['term','rank'])
    words = (ranking.sort_values('rank', ascending = False))
    print ("\n\nWords head : \n", words.head(7))


if __name__ == '__main__':
    main()
