from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2

data_folder = 'yelpNgrams'
dataset = load_files( data_folder, shuffle=False)
print(dataset.target_names)
docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)

#--- Create train matrix ----
#count_vect = CountVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2), analyzer=u'word')
#count_vect = CountVectorizer(lowercase=True)
#X_train_counts = count_vect.fit_transform(dataset_train.data)
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

##### N-GRAMS WITHOUT FREQUENCIES
'''
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), max_df=0.7, min_df=2, max_features=30)
X_train = vectorizer.fit_transform(docs_train)
X_test = vectorizer.transform(docs_test)
print(vectorizer.get_feature_names())
print()
'''

##### N-GRAMS WITH FREQUENCIES
ngram_vectorizer = CountVectorizer(stop_words='english', lowercase=True, ngram_range=(3, 3), analyzer=u'word',max_features=30, max_df=0.90, min_df=3)
counts = ngram_vectorizer.fit_transform(docs_train)
vocab = ngram_vectorizer.vocabulary_
count_values = counts.toarray().sum(axis=0)
counts = sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
print(counts)

