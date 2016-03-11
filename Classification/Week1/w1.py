import string
import sys
print(sys.version)

import numpy as np
import sframe
products = sframe.SFrame('amazon_baby.gl/')
products
products[269]

#Remove punctuation using Python's built-in string functionality.
products['review_clean'] = products['review'].apply(lambda text: text.translate(None, string.punctuation))

#We will ignore all reviews with rating = 3, since they tend to have a neutral sentiment.
products = products[products['rating'] != 3]
print len(products)

# assign reviews with a rating of 4 or higher to be positive reviews, while the ones with rating of 2 or lower are negative
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
products

#Split data into training and test sets
train_data, test_data = products.random_split(.8, seed=1)
print len(train_data)
print len(test_data)

#compute the word count for each word that appears in the reviews
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
# Use this token pattern to keep single-letter words
# First, learn vocabulary from the training data and assign columns to words
# Then convert the training data into a sparse matrix
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
# Second, convert the test data into a sparse matrix, using the same word-column mapping
test_matrix = vectorizer.transform(test_data['review_clean'])

#Train a sentiment classifier with logistic regression
from sklearn import linear_model 
sentiment_model = linear_model.LogisticRegression()
sentiment_model.fit(train_matrix, train_data['sentiment'])

q1 = (sentiment_model.coef_ >= 0).sum()

#Making predictions with logistic regression
sample_test_data = test_data[10:13]
print sample_test_data

sample_test_data[0]['review']
sample_test_data[1]['review']

sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
scores = sentiment_model.decision_function(sample_test_matrix)
print scores

sample_test_predict = sentiment_model.predict(sample_test_matrix)
sample_test_predict_proba = sentiment_model.predict_proba(sample_test_matrix)
sample_test_predict_1 = 1/(1+np.exp(-scores))
q2 = np.where(sample_test_predict_1 == min(sample_test_predict_1))[0][0]+1

#Find the most positive (and negative) review on all data
test_matrix = vectorizer.transform(test_data['review_clean'])
test_predict = sentiment_model.predict(test_matrix)
test_predict_proba = sentiment_model.predict_proba(test_matrix)

#top20 = test_predict_proba[:,1].argsort()[-20:]
#test_predict_proba[top20]
#test_data[top20]

#top 20 pos and neg reivews
test_data['proba_pos'] = test_predict_proba[:,1]
test_data['proba_neg'] = test_predict_proba[:,0]

q3 = test_data.topk('proba_pos', k=20)['name']
q4 = test_data.topk('proba_neg', k=20, reverse=True)['name']

#Compute accuracy of the classifier
test_data['predict'] = test_predict
count_correct = (test_data['predict'] == test_data['sentiment']).sum()
q5 = float(count_correct)/test_data.num_rows()
#sentiment_model.score(test_matrix, np.array(test_data['sentiment']))

#Learn another classifier with fewer words
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']
      
vectorizer_word_subset = CountVectorizer(vocabulary=significant_words)
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])

simple_model = linear_model.LogisticRegression()
simple_model.fit(train_matrix_word_subset, train_data['sentiment'])

simple_model_coef_table = sframe.SFrame({'word':significant_words,
                                         'coefficient':simple_model.coef_.flatten()})

q6 = simple_model.coef_[simple_model.coef_ >= 0].size

q7a = sentiment_model.score(train_matrix, np.array(train_data['sentiment']))
q7b = simple_model.score(train_matrix_word_subset, np.array(train_data['sentiment']))

q8a = sentiment_model.score(test_matrix, np.array(test_data['sentiment']))
q8b = simple_model.score(test_matrix_word_subset, np.array(test_data['sentiment']))

count_majority = test_data['sentiment'].sum()
q9 = float(count_majority)/test_data.num_rows()
