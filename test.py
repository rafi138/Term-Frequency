from tfidf import tfIDF

dataset = ['my name is tfidf', 'her name is feature extraction','his name is fancy']

tfidf_object = tfIDF(dataset)
test_str='name feature extraction'
idx=tfidf_object.search(test_str)[0][0]
print(dataset[idx])
