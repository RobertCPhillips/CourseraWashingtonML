import sframe
products = sframe.SFrame('amazon_baby.gl/')

#Remove punctuation using Python's built-in string functionality.
import string
products['review_clean'] = products['review'].apply(lambda text: text.translate(None, string.punctuation))

#We will ignore all reviews with rating = 3, since they tend to have a neutral sentiment.
products = products[products['rating'] != 3]

# assign reviews with a rating of 4 or higher to be positive reviews, while the ones with rating of 2 or lower are negative
pos_sentiment = +1
neg_sentiment = -1
products['sentiment'] = products['rating'].apply(lambda rating : pos_sentiment if rating > 3 else neg_sentiment)

#Split data into training and test sets
train_data, test_data = products.random_split(.8, seed=1)
print len(train_data)
print len(test_data)

test_data_sentiment_np = test_data['sentiment'].to_numpy()

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
sentiment_model_pred = sentiment_model.predict(test_matrix)

#Compute the accuracy on the test set
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true=test_data_sentiment_np, y_pred=sentiment_model_pred)
print "Test Accuracy: %s" % accuracy

#Baseline: Majority class prediction
baseline = len(test_data[test_data['sentiment'] == pos_sentiment])/float(len(test_data))
print "Baseline accuracy (majority class classifier): %s" % baseline

#Quiz question 1: Using accuracy as the evaluation metric, was our logistic 
# regression model better than the baseline (majority class classifier)?
q1 = "yes"

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cmat = confusion_matrix(y_true=test_data_sentiment_np,
                        y_pred=sentiment_model_pred,
                        labels=sentiment_model.classes_)    # use the same order of class as the LR model.

# Print out the confusion matrix.
print ' target_label | predicted_label | count '
print '--------------+-----------------+-------'
for i, target_label in enumerate(sentiment_model.classes_):
    for j, predicted_label in enumerate(sentiment_model.classes_):
        print '{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[i,j])


#Quiz Question 2: How many predicted values in the test set are false positives?
q2 = 1451

#Computing the cost of mistakes

#Suppose you know the costs involved in each kind of mistake:
#    $100 for each false positive.
#    $1 for each false negative.
#    Correctly classified reviews incur no cost.

#Quiz Question 3: Given the stipulation, what is the cost associated with the 
# logistic regression classifier's performance on the test set?
q3 = 100.*1451 + 1.*806

#Computing precision and recall
from sklearn.metrics import precision_score
precision = precision_score(y_true=test_data_sentiment_np, 
                            y_pred=sentiment_model_pred)
print "Precision on test data: %s" % precision

#Quiz Question 4: Out of all reviews in the test set that are predicted to be 
# positive, what fraction of them are false positives? 
q4 = 1. - precision

#Quiz Question 5: Based on what we learned in lecture, if we wanted to reduce 
# this fraction of false positives to be below 3.5%, we would: (see quiz)
q5 = "increase threshold for positive review"

from sklearn.metrics import recall_score
recall = recall_score(y_true=test_data_sentiment_np,
                      y_pred=sentiment_model_pred)
print "Recall on test data: %s" % recall

#Quiz Question 6: What fraction of the positive reviews in the test_set were 
# correctly predicted as positive by the classifier?
q6 = recall

#Quiz Question 7: What is the recall value for a classifier that predicts +1 
# for all data points in the test_data?
q7 = 1

#--------------------------------------
#Precision-recall tradeoff
#--------------------------------------

'''
Returns an array, where each element is set to +1 or -1 depending whether the 
corresponding probability exceeds threshold.

  probabilities: an SArray of probability values
  threshold: a float between 0 and 1
'''
def apply_threshold(probabilities, threshold):
    return probabilities.apply(lambda p: +1 if p > threshold else -1)

sentiment_model_proba = sentiment_model.predict_proba(test_matrix)
sentiment_model_proba_p1_sf = sframe.SArray(sentiment_model_proba[:,1])

pred_p50 = apply_threshold(sentiment_model_proba_p1_sf, .5)
pred_p90 = apply_threshold(sentiment_model_proba_p1_sf, .9)

pred_p50_len = len(pred_p50[pred_p50 == 1L])
pred_p90_len = len(pred_p90[pred_p90 == 1L])

#Quiz question 8: What happens to the number of positive predicted reviews as 
# the threshold increased from 0.5 to 0.9?
q8 = "they went down" 

#Compute precision and recall for threshold values 0.5 and 0.9.
prec_90 = precision_score(y_true=test_data_sentiment_np, y_pred=pred_p90.to_numpy())
prec_50 = precision_score(y_true=test_data_sentiment_np, y_pred=pred_p50.to_numpy())
recall_90 = recall_score(y_true=test_data_sentiment_np, y_pred=pred_p90.to_numpy())
recall_50 = recall_score(y_true=test_data_sentiment_np, y_pred=pred_p50.to_numpy())
        
#Quiz Question 9 (variant 1): Does the precision increase with a higher threshold?
q9a = "Precision 50: %s  Precision 90: %s" % (prec_50,prec_90)

#Quiz Question 9 (variant 2): Does the recall increase with a higher threshold?
q9b = "Recall 50: %s  Recall 90: %s" % (recall_50,recall_90)

#-------------------------------------------------
#Precision-recall curve
#-------------------------------------------------
import numpy as np
threshold_values = np.linspace(0.5, 1, num=100)
print threshold_values

precision_all = []
recall_all = []

for i in threshold_values:
    predictions = apply_threshold(sentiment_model_proba_p1_sf, i).to_numpy()
    p = precision_score(y_true=test_data_sentiment_np, y_pred=predictions)
    r = recall_score(y_true=test_data_sentiment_np, y_pred=predictions)
    
    precision_all.append(p)
    recall_all.append(r)
    
import matplotlib.pyplot as plt

def plot_pr_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.rcParams.update({'font.size': 16})

plot_pr_curve(precision_all[:99], recall_all[:99], 'Precision recall curve (all)')

#Quiz Question 10: Among all the threshold values tried, what is the smallest 
# threshold value that achieves a precision of 96.5% or better? Round your 
# answer to 3 decimal places.
q10 = [threshold_values[i] for i, j in enumerate(precision_all) if j >= .965][0]

#Quiz Question 11: Using threshold = 0.98, how many false negatives do we get 
# on the test_data? This is the number of false negatives (i.e the number of 
# reviews to look at when not needed) that we have to deal with using this classifier.
q11_predictions = apply_threshold(sentiment_model_proba_p1_sf, .98).to_numpy()

q11_cmat = confusion_matrix(y_true=test_data_sentiment_np,
                        y_pred=q11_predictions,
                        labels=sentiment_model.classes_)    # use the same order of class as the LR model.

# Print out the confusion matrix.
print ' target_label | predicted_label | count '
print '--------------+-----------------+-------'
for i, target_label in enumerate(sentiment_model.classes_):
    for j, predicted_label in enumerate(sentiment_model.classes_):
        print '{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, q11_cmat[i,j])

q11 = 8293

#------------------------------------------------
#Evaluating specific search terms
#------------------------------------------------

#select all the reviews for all products with the word 'baby' and predict
baby_reviews = test_data[test_data['name'].apply(lambda x: 'baby' in x.lower())]

baby_matrix = vectorizer.transform(baby_reviews['review_clean'])
baby_proba = sentiment_model.predict_proba(baby_matrix)
baby_proba_p1_sf = sframe.SArray(baby_proba[:,1])

baby_sentiment_np = baby_reviews['sentiment'].to_numpy()

precision_all = []
recall_all = []

for i in threshold_values:
    predictions = apply_threshold(baby_proba_p1_sf, i).to_numpy()
    p = precision_score(y_true=baby_sentiment_np, y_pred=predictions)
    r = recall_score(y_true=baby_sentiment_np, y_pred=predictions)
    
    precision_all.append(p)
    recall_all.append(r)

plot_pr_curve(precision_all[:99], recall_all[:99], 'Precision recall curve (baby)')

#Quiz Question 12: Among all the threshold values tried, what is the smallest 
# threshold value that achieves a precision of 96.5% or better for the reviews 
# of data in baby_reviews? Round your answer to 3 decimal places.
q12 = [threshold_values[i] for i, j in enumerate(precision_all) if j >= .965][0]

#Quiz Question 13: Is this threshold value smaller or larger than the threshold 
# used for the entire dataset to achieve the same specified precision of 96.5%?
q3 = "larger"
