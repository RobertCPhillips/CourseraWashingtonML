import sys
print(sys.version)

import sframe
products = sframe.SFrame('amazon_baby_subset.gl/')
products
products.num_rows()

#The sentiment column corresponds to the class label with +1 indicating a review with positive sentiment and -1 for negative sentiment.
#The name column indicates the name of the product.
#The review column contains the text of the user's review.
#The rating column contains the user's rating.

products.head(10)['name']

print '# of positive reviews =', len(products[products['sentiment']==1])
print '# of negative reviews =', len(products[products['sentiment']==-1])

#=============================================
#Apply Text Cleaning on the Review Data
#=============================================

# Reads the list of most frequent words
import json
with open('important_words.json', 'r') as f:
    important_words = json.load(f)

important_words = [str(s) for s in important_words]
important_words

#Remove punctuation using Python's built-in string functionality.
import string
products['review_clean'] = products['review'].apply(lambda text: text.translate(None, string.punctuation))

#Create a column for each word in important_words which keeps a count of the number of times the respective word occurs in the review text.
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))

#==question 1 - How many reviews contain the word perfect? 
q1 = len(products[products['perfect'] > 0]) #2955

#=============================================
#Convert SFrame to NumPy array
#=============================================
import numpy as np

'''
input parameters:
  data_sframe: a data frame to be converted
  features: a list of string, containing the names of the columns that are used as features.
  label: a string, containing the name of the single column that is used as class labels.

output:
  a 2D array for features
  a 1D array for class labels
'''
def get_numpy_data(data_sframe, features, label):
    data_sframe['intercept'] = 1
    features = ['intercept'] + features
    features_sframe = data_sframe[features]
    feature_matrix = features_sframe.to_numpy()
    label_sarray = data_sframe[label]
    label_array = label_sarray.to_numpy()
    return(feature_matrix, label_array)

feature_matrix, sentiment = get_numpy_data(products, important_words, 'sentiment')

#==question 2 - How many features are there in the feature_matrix? 
q2 = feature_matrix.shape[1]

#==question 3 - Assuming that the intercept is present, how does the 
#               number of features in feature_matrix relate to the number 
#               of features in the logistic regression model?
q3 = 'intercepts adds a term, or they are the same??'


#=============================================
#Estimating conditional probability with link function
#=============================================

'''
produces probablistic estimate for P(y_i = +1 | x_i, w).
estimate ranges between 0 and 1.
'''
def predict_probability(feature_matrix, coefficients):
    # Take dot product of feature_matrix and coefficients  
    score = feature_matrix.dot(coefficients)
    
    # Compute P(y_i = +1 | x_i, w) using the link function
    predictions = 1./(1.+np.exp(-score))
    
    # return predictions
    return predictions

#test
dummy_feature_matrix = np.array([[1.,2.,3.], [1.,-1.,-1]])
dummy_coefficients = np.array([1., 3., -1.])

correct_scores      = np.array( [ 1.*1. + 2.*3. + 3.*(-1.),          1.*1. + (-1.)*3. + (-1.)*(-1.) ] )
correct_predictions = np.array( [ 1./(1+np.exp(-correct_scores[0])), 1./(1+np.exp(-correct_scores[1])) ] )

print 'The following outputs must match '
print '------------------------------------------------'
print 'correct_predictions           =', correct_predictions
print 'output of predict_probability =', predict_probability(dummy_feature_matrix, dummy_coefficients)


#=============================================
#Compute derivative of log likelihood with respect to a single coefficient
#=============================================
'''
errors: vector whose i-th value contains the difference between label is +1 and prob +1 given score
feature: vector whose i-th value contains score
'''
def feature_derivative(errors, feature):     
    # Compute the dot product of errors and feature
    #note: using dot product becuase of sum of products
    derivative = errors.dot(feature)
    
    # Return the derivative
    return derivative

'''
'''
def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))
    
    # Simple check to prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]
    
    lp = np.sum((indicator-1)*scores - logexp)
    return lp

#test
dummy_feature_matrix = np.array([[1.,2.,3.], [1.,-1.,-1]])
dummy_coefficients = np.array([1., 3., -1.])
dummy_sentiment = np.array([-1, 1])

correct_indicators  = np.array( [ -1==+1,                                       1==+1 ] )
correct_scores      = np.array( [ 1.*1. + 2.*3. + 3.*(-1.),                     1.*1. + (-1.)*3. + (-1.)*(-1.) ] )
correct_first_term  = np.array( [ (correct_indicators[0]-1)*correct_scores[0],  (correct_indicators[1]-1)*correct_scores[1] ] )
correct_second_term = np.array( [ np.log(1. + np.exp(-correct_scores[0])),      np.log(1. + np.exp(-correct_scores[1])) ] )

correct_ll          =      sum( [ correct_first_term[0]-correct_second_term[0], correct_first_term[1]-correct_second_term[1] ] ) 

print 'The following outputs must match '
print '------------------------------------------------'
print 'correct_log_likelihood           =', correct_ll
print 'output of compute_log_likelihood =', compute_log_likelihood(dummy_feature_matrix, dummy_sentiment, dummy_coefficients)

#=============================================
#Taking gradient steps
#=============================================

'''
feature_matrix: 2D array of features
sentiment: 1D array of class labels
initial_coefficients: 1D array containing initial values of coefficients
step_size: a parameter controlling the size of the gradient steps
max_iter: number of iterations to run gradient ascent
'''
def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    
    for itr in xrange(max_iter):
        # Predict P(y_i = +1|x_i,w) using your predict_probability() function
        # YOUR CODE HERE
        predictions = predict_probability(feature_matrix, coefficients)
        
        # Compute indicator value for (y_i = +1)
        indicator = (sentiment==+1)
        
        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in xrange(len(coefficients)): # loop over each coefficient
            
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j].
            # Compute the derivative for coefficients[j]. Save it in a variable called derivative
            # YOUR CODE HERE
            derivative = feature_derivative(errors, feature_matrix[:,j])
            
            # add the step size times the derivative to the current coefficient
            ## YOUR CODE HERE
            coefficients[j] += derivative * step_size
        
        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp)
                
    return coefficients

coefficients = logistic_regression(feature_matrix, sentiment, initial_coefficients=np.zeros(194),
                                   step_size=1e-7, max_iter=301)


#==question 4 - As each iteration of gradient ascent passes, does the log likelihood increase or decrease?
q4 = 'increased??'

#=============================================
#Predicting sentiments
#=============================================

#compute scores
scores = np.dot(feature_matrix, coefficients)

#compute class prediction
predicts = np.array([+1 if score > 0 else -1 for score in scores])


#==question 5 - How many reviews were predicted to have positive sentiment?
#correct = np.sum(predicts == products['sentiment']) #39334
q5 = np.sum(predicts == +1) #25126

#=============================================
#Measuring accuracy
#=============================================
num_mistakes = np.sum(predicts != products['sentiment'])
accuracy = 1. - (float(num_mistakes) / float(len(predicts)))
print "-----------------------------------------------------"
print '# Reviews   correctly classified =', len(products) - num_mistakes
print '# Reviews incorrectly classified =', num_mistakes
print '# Reviews total                  =', len(products)
print "-----------------------------------------------------"
print 'Accuracy = %.2f' % accuracy

#==question 6 -  What is the accuracy of the model on predictions?
q6 = accuracy #.75


#=============================================
#Which words contribute most to positive & negative sentiments?
#=============================================
coefficients2 = list(coefficients[1:]) # exclude intercept
word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(important_words, coefficients2)]
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=True)
q7 = word_coefficient_tuples[:10]
q8 = word_coefficient_tuples[-10:]






