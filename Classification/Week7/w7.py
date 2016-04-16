import sframe
products = sframe.SFrame('amazon_baby_subset.gl/')
products['review'] = products['review'].fillna('')

# Reads the list of most frequent words
import json
with open('important_words.json', 'r') as f:
    important_words = json.load(f)

important_words = [str(s) for s in important_words]

#Remove punctuation using Python's built-in string functionality.
import string
products['review_clean'] = products['review'].apply(lambda text: text.translate(None, string.punctuation))

#Create a column for each word in important_words which keeps a count of the number of times the respective word occurs in the review text.
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))

#Split data into training and validation sets
train_data, validation_data = products.random_split(.9, seed=1)

#Convert SFrame to NumPy array
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
    
#def get_numpy_data(data_sframe, label):
#    label_sarray = data_sframe[label]
#    label_array = label_sarray.to_numpy()
#    
#    data_sframe.remove_column(label)
#    feature_matrix = data_sframe.to_numpy()
#    
#    return(feature_matrix, label_array)

feature_matrix_train, sentiment_train = get_numpy_data(train_data, important_words, 'sentiment')
feature_matrix_valid, sentiment_valid = get_numpy_data(validation_data, important_words, 'sentiment')

#Quiz question 2: How does the changing the solver to stochastic gradient ascent affect the number of features?
q2 = "it doesn't"

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

def compute_avg_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))
    
    # Simple check to prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]
    
    lp = np.sum((indicator-1)*scores - logexp) / float(len(feature_matrix))
    return lp

#Quiz question 3: How are the functions ll(w) and ll_A(w) related?
q3 = "ll(w) = N * ll_A(w)"

#test 1 data point
j = 1                        # Feature number
i = 10                       # Data point number
coefficients = np.zeros(194) # A point w at which we are computing the gradient.

predictions = predict_probability(feature_matrix_train[i:i+1,:], coefficients)
indicator = (sentiment_train[i:i+1]==+1)

errors = indicator - predictions
gradient_single_data_point = feature_derivative(errors, feature_matrix_train[i:i+1,j])
print "Gradient single data point: %s" % gradient_single_data_point
print "           --> Should print 0.0"

#Quiz Question 4: The code block above computed the derivative for j = 1 and i = 10. 
# Is this a scalar or a 194-dimensional vector?
q4 = "scaler"

#test batch of points
j = 1                        # Feature number
i = 10                       # Data point start
B = 10                       # Mini-batch size
coefficients = np.zeros(194) # A point w at which we are computing the gradient.

predictions = predict_probability(feature_matrix_train[i:i+B,:], coefficients)
indicator = (sentiment_train[i:i+B]==+1)

errors = indicator - predictions
gradient_mini_batch = feature_derivative(errors, feature_matrix_train[i:i+B,j])
print "Gradient mini-batch data points: %s" % gradient_mini_batch
print "                --> Should print 1.0"

#Quiz Question 5: The code block above computed for j = 10, i = 10, and B = 10. 
# Is this a scalar or a 194-dimensional vector?
q5 = "scaler"

#Quiz Question 6: For what value of B is the term the same as the full gradient?
q6 = "B = N"

def logistic_regression_SG(feature_matrix, sentiment, initial_coefficients, step_size, batch_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    log_likelihood_all = []
    data_size = len(feature_matrix)
    
    # Shuffle the data before starting
    np.random.seed(seed=1)
    permutation = np.random.permutation(data_size)
    feature_matrix = feature_matrix[permutation,:]
    sentiment = sentiment[permutation]
    
    i = 0 # index of current batch
    # Do a linear scan over data
    for itr in xrange(max_iter):
        # Predict P(y_i = +1|x_i,w) using your predict_probability() function
        predictions = predict_probability(feature_matrix[i:i+batch_size,:], coefficients)
        
        # Compute indicator value for (y_i = +1)
        indicator = (sentiment[i:i+batch_size]==+1)
        
        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in xrange(len(coefficients)): # loop over each coefficient
            
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j].
            # Compute the derivative for coefficients[j]. Save it in a variable called derivative
            derivative = feature_derivative(errors, feature_matrix[i:i+batch_size:,j])
            
            # add the step size times the derivative to the current coefficient
            coefficients[j] += derivative * step_size / float(batch_size)
        
        # Checking whether log likelihood is increasing
        # Print the log likelihood over the *current batch*
        lp = compute_avg_log_likelihood(feature_matrix[i:i+batch_size,:], sentiment[i:i+batch_size],
                                        coefficients)
        log_likelihood_all.append(lp)

        if itr <= 15 or (itr <= 1000 and itr % 100 == 0) or (itr <= 10000 and itr % 1000 == 0) \
         or itr % 10000 == 0 or itr == max_iter-1:
            print 'Iteration %*d: Average log likelihood (of data points  [%0*d:%0*d]) = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, \
                 int(np.ceil(np.log10(data_size))), i, \
                 int(np.ceil(np.log10(data_size))), i+batch_size, lp)
                 
        # if we made a complete pass over data, shuffle and restart
        i += batch_size
        if i+batch_size > data_size:
            permutation = np.random.permutation(data_size)
            feature_matrix = feature_matrix[permutation,:]
            sentiment = sentiment[permutation]
            i = 0 
                
    return coefficients, log_likelihood_all

#Quiz Question 7: For what value of batch size B above is the stochastic gradient 
# ascent function logistic_regression_SG act as a standard gradient ascent algorithm?
q7 = len(train_data)

#16 - run stochastic gradient ascent over the feature_matrix_train for 10 iterations using:
coefficients1, ll1 = logistic_regression_SG(feature_matrix_train, sentiment_train, initial_coefficients=np.zeros(194),
                                   step_size=5e-1, batch_size = 1, max_iter=10)

#17 - run batch gradient ascent over the feature_matrix_train for 200 iterations using:
coefficients2, ll2 = logistic_regression_SG(feature_matrix_train, sentiment_train, initial_coefficients=np.zeros(194),
                                   step_size=5e-1, batch_size = len(feature_matrix_train), max_iter=200)

#plot log likelihoods
import matplotlib.pyplot as plt
plt.plot(range(len(ll1)), ll1, linewidth=4.0, label='Log Likelihood Batch Size = 1')
plt.plot(range(len(ll2)), ll2, linewidth=4.0, label='Log Likelihood Batch Size = N')

#Quiz Question 8. When you set batch_size = 1, as each iteration passes, how does 
# the average log likelihood in the batch change?
q8 = "Fluctuates"

#Quiz Question 9. When you set batch_size = len(train_data), as each iteration passes, 
# how does the average log likelihood in the batch change?
q9 = "It increases smoothly"

#Quiz Question 10. Suppose that we run stochastic gradient ascent with a batch 
# size of 100. How many gradient updates are performed at the end of two passes 
# over a dataset consisting of 50000 data points?
q10 = 50000/100*2

#19 - stochastic gradient ascent for 10 passes
batch_size = 100
num_passes = 10
num_iterations = num_passes * int(len(feature_matrix_train)/batch_size)
coefficients3, ll3 = logistic_regression_SG(feature_matrix_train, sentiment_train, initial_coefficients=np.zeros(194),
                                   step_size=1e-1, batch_size = batch_size, max_iter=num_iterations)

plt.plot(range(len(ll3)), ll3, linewidth=4.0, label='Log Likelihood Batch Size = 100')

'''
To make a fair comparison betweeen stochastic gradient ascent and batch gradient ascent, 
we measure the average log likelihood as a function of the number of passes =

  (# of data points) / (size of the data set)
  
This function generates a plot of the average log likelihood as a function of the 
number of passes. 

The parameters are:

    log_likelihood_all, the list of average log likelihood over time
    len_data, number of data points in the training set
    batch_size, size of each mini-batch
    smoothing_window, a parameter for computing moving averages
'''
def make_plot(log_likelihood_all, len_data, batch_size, smoothing_window=1, label=''):
    plt.rcParams.update({'figure.figsize': (9,5)})
    log_likelihood_all_ma = np.convolve(np.array(log_likelihood_all), \
                                        np.ones((smoothing_window,))/smoothing_window, mode='valid')

    plt.plot(np.array(range(smoothing_window-1, len(log_likelihood_all)))*float(batch_size)/len_data,
             log_likelihood_all_ma, linewidth=4.0, label=label)
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.xlabel('# of passes over data')
    plt.ylabel('Average log likelihood per data point')
    plt.legend(loc='lower right', prop={'size':14})

#batch gradient ascent = ll2
make_plot(ll2, len(feature_matrix_train), len(feature_matrix_train), smoothing_window=1, label='batch gradient ascent for 200 iter.')

make_plot(ll3, len_data=len(feature_matrix_train), batch_size=100, label='stochastic gradient, step_size=1e-1')
make_plot(ll3, len_data=len(feature_matrix_train), batch_size=100, smoothing_window=30, label='stochastic gradient, step_size=1e-1')
          
#21 - stochastic gradient ascent for 200 passes
batch_size = 100
num_passes = 200
num_iterations = num_passes * int(len(feature_matrix_train)/batch_size)
coefficients4, ll4 = logistic_regression_SG(feature_matrix_train, sentiment_train, initial_coefficients=np.zeros(194),
                                   step_size=1e-1, batch_size = batch_size, max_iter=num_iterations)

make_plot(ll4, len(feature_matrix_train), 100, smoothing_window=30, label='stochastic gradient ascent for 200 iter.')


#Quiz Question 11: In the figure above, how many passes does batch gradient ascent 
# need to achieve a similar log likelihood as stochastic gradient ascent?
q11 = "150 passes or more"

#Explore the effects of step sizes on stochastic gradient ascent
batch_size = 100
num_passes = 10
num_iterations = num_passes * int(len(feature_matrix_train)/batch_size)

coefficients_sgd = {}
log_likelihood_sgd = {}
step_sizes = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
for ss in step_sizes:
    coefficients_sgd[ss], log_likelihood_sgd[ss] = logistic_regression_SG(feature_matrix_train, sentiment_train, 
                              initial_coefficients=np.zeros(194),
                              step_size=ss, batch_size = batch_size, max_iter=num_passes)

for step_size in step_sizes:
    make_plot(log_likelihood_sgd[step_size], len_data=len(train_data), batch_size=100,
              smoothing_window=30, label='step_size=%.1e'%step_size)

for step_size in step_sizes[0:6]:
    make_plot(log_likelihood_sgd[step_size], len_data=len(train_data), batch_size=100,
              smoothing_window=30, label='step_size=%.1e'%step_size)

#Quiz Question 12: Which of the following is the worst step size? Pick the step size that results in the lowest log likelihood in the end.

#Quiz Question 13: Which of the following is the best step size? Pick the step size that results in the highest log likelihood in the end.
              