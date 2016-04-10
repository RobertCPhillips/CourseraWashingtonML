# -*- coding: utf-8 -*-
import sframe
loans = sframe.SFrame('lending-club-data.gl/')

safe_loan_label = +1
risky_loan_label = -1
target = 'safe_loans'

#re-assign the target to have +1 as a safe (good) loan, and -1 as a risky (bad) loan.
loans[target] = loans['bad_loans'].apply(lambda x : safe_loan_label if x==0 else risky_loan_label)
loans.remove_column('bad_loans')

features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]

#extract feature and target
loans = loans[features + [target]]


#-------------------------------------------
#Sample data to balance classes
#-------------------------------------------
safe_loans_raw = loans[loans[target] == safe_loan_label]
risky_loans_raw = loans[loans[target] == risky_loan_label]

# Since there are less risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(percentage, seed = 1)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)

#-------------------------------------------
#One-hot encoding
#-------------------------------------------
categorical_variables = []
for feat_name, feat_type in zip(loans_data.column_names(), loans_data.column_types()):
    if feat_type == str:
        categorical_variables.append(feat_name)

for feature in categorical_variables:
    loans_data_one_hot_encoded = loans_data[feature].apply(lambda x: {x: 1})
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature)

    # Change None's to 0's
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)

    loans_data.remove_column(feature)
    loans_data.add_columns(loans_data_unpacked)

features = loans_data.column_names() # get features with encoding
features.remove(target)  # Remove the response variable

#-------------------------------------------
#Split data into training and validation
#-------------------------------------------
train_data, validation_data = loans_data.random_split(.8, seed=1)

#-------------------------------------------
#Function to pick best feature to split on
#-------------------------------------------
def intermediate_node_weighted_mistakes(labels_in_node, data_weights):
    # Sum the weights of all entries with label +1
    total_weight_positive = data_weights[labels_in_node == safe_loan_label].sum()
    
    # Weight of mistakes for predicting all -1's is equal to the sum above
    ### YOUR CODE HERE
    weighted_mistakes_all_negative = total_weight_positive
    
    # Sum the weights of all entries with label -1
    ### YOUR CODE HERE
    total_weight_negative = data_weights[labels_in_node == risky_loan_label].sum()
    
    # Weight of mistakes for predicting all +1's is equal to the sum above
    ### YOUR CODE HERE
    weighted_mistakes_all_positive = total_weight_negative
    
    # Return the tuple (weight, class_label) representing the lower of the two weights
    #    class_label should be an integer of value +1 or -1.
    # If the two weights are identical, return (weighted_mistakes_all_positive,+1)
    ### YOUR CODE HERE
    if (weighted_mistakes_all_negative < weighted_mistakes_all_positive):
        return (weighted_mistakes_all_negative, risky_loan_label)
    else:
        return (weighted_mistakes_all_positive, safe_loan_label)

#Quiz Question 1: If we set the weights a=1 for all data points, how is the 
# weight of mistakes WM(a,ŷ) related to the classification error? 
q1 = "it is the # of mistakes"

'''
The function will loop through the list of possible features, and consider 
splitting on each of them. It will calculate the weighted error of each 
split and return the feature that had the smallest weighted error when 
split on.
'''
def best_splitting_feature(data, features, target, data_weights):
    
    best_feature = None # Keep track of the best feature 
    best_error = float('+inf') # Keep track of the best error so far 
    data_weights_total = float(data_weights.sum())
    
    # Loop through each feature to consider splitting on that feature
    for feature in features:
        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        
        # The right split will have all data points where the feature value is 1
        right_split =  data[data[feature] == 1]
            
        # Calculate the number of misclassified examples in the left split.
        left_weights = data_weights[data[feature] == 0]
        left_weighted_mistakes, left_class = intermediate_node_weighted_mistakes(left_split[target], left_weights)

        # Calculate the number of misclassified examples in the right split.
        right_weights = data_weights[data[feature] == 1]
        right_weighted_mistakes, right_class = intermediate_node_weighted_mistakes(right_split[target], right_weights)
            
        # Compute the classification error of this split.
        error = float(left_weighted_mistakes + right_weighted_mistakes) / data_weights_total
        #print "the error is (%s + %s) / %s = %s" % (left_weighted_mistakes, right_weighted_mistakes, data_weights_total, error)
        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        if error < best_error:
            best_error = error
            best_feature = feature
    
    return best_feature # Return the best feature we found

'''
Each node in the decision tree is represented as a dictionary which contains the 
following keys and possible values:
    
'is_leaf'            : True/False.
'prediction'         : Prediction at the leaf node.
'left'               : (dictionary corresponding to the left tree).
'right'              : (dictionary corresponding to the right tree).
'splitting_feature'  : The feature that this node splits on.}
'''
def create_leaf(target_values, data_weights):    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True } 
   
    # Computed weight of mistakes.
    # Store the predicted class (1 or -1) in leaf['prediction']
    weighted_error, best_class = intermediate_node_weighted_mistakes(target_values, data_weights)
    leaf['prediction'] = best_class

    # Return the leaf node
    return leaf

'''
Learns the decision tree recursively and implements 3 stopping conditions:

1. Stopping condition 1: The error is zero.
2. Stopping condition 2: No more features to split on.
3. Stopping condition 3: The max_depth of the tree. 
   
   By not letting the tree grow beyond these thresholds, we will save 
   computational effort in the learning process and reduce overfitting. 
'''
def decision_tree_create(data, features, target, data_weights, current_depth = 0, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))

    # Stopping condition 1
    # (Check if there are mistakes at current node.
    # Recall you wrote a function intermediate_node_num_mistakes to compute this.)
    if intermediate_node_weighted_mistakes(target_values, data_weights) <= 1e-15:
        print "Stopping condition 1 reached."     
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values, data_weights)
    
    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if remaining_features == []:
        print "Stopping condition 2 reached."    
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values, data_weights)
    
    # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth:
        print "Reached maximum depth. Stopping for now."
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values, data_weights)

    # Find the best splitting feature (recall the function best_splitting_feature implemented above)
    splitting_feature = best_splitting_feature(data, remaining_features, target, data_weights)
    #print "Best Split Feature %s " % (splitting_feature)
    remaining_features.remove(splitting_feature)
        
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]

    left_data_weights = data_weights[data[splitting_feature] == 0]
    right_data_weights = data_weights[data[splitting_feature] == 1]
    
    print "Split on feature %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print "Creating leaf node."
        return create_leaf(left_split[target], left_data_weights)
    if len(right_split) == len(data):
        print "Creating leaf node."
        return create_leaf(right_split[target], right_data_weights)
        
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, left_data_weights, current_depth + 1, max_depth)        
    right_tree = decision_tree_create(right_split, remaining_features, target, right_data_weights, current_depth + 1, max_depth)

    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}

def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])

#--------------------------------------------------
#Making predictions with a weighted decision tree
#--------------------------------------------------
def classify(tree, x, annotate = False):   
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate: 
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction'] 
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate: 
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)

def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x))
    
    # Once you've made the predictions, calculate the classification error and return it
    return float((prediction != data[target]).sum()) / data.num_rows()


#--------------------------------------------------
#Example: Training a weighted decision tree 
#--------------------------------------------------

# Assign weights
example_data_weights = sframe.SArray([1.] * 10 + [0.]*(len(train_data) - 20) + [1.] * 10)

# Train a weighted decision tree model.
small_data_decision_tree_subset_20 = decision_tree_create(train_data, features, target,
                         example_data_weights, max_depth=2)

evaluate_classification_error(small_data_decision_tree_subset_20, train_data)

subset_20 = train_data.head(10).append(train_data.tail(10))
evaluate_classification_error(small_data_decision_tree_subset_20, subset_20)

small_data_decision_tree_subset_20_only = decision_tree_create(subset_20, features, target,
                         example_data_weights.head(10).append(example_data_weights.tail(10)), max_depth=2)

#Quiz Question : Will you get the same model as small_data_decision_tree_subset_20 
# if you trained a decision tree with only the 20 data points with non-zero 
# weights from the set of points in subset_20?
q2 = "no"

#--------------------------------------------------
#Implementing Adaboost on decision stumps
#--------------------------------------------------
from math import log
from math import exp

def adaboost_with_tree_stumps(data, features, target, num_tree_stumps):
    # start with unweighted data
    alpha = sframe.SArray([1.]*len(data))
    weights = []
    tree_stumps = []
    target_values = data[target]
    
    for t in xrange(num_tree_stumps):
        print '====================================================='
        print 'Adaboost Iteration %d' % t
        print '====================================================='        
        # Learn a weighted decision tree stump. Use max_depth=1
        tree_stump = decision_tree_create(data, features, target, data_weights=alpha, max_depth=1)
        tree_stumps.append(tree_stump)
        
        # Make predictions
        predictions = data.apply(lambda x: classify(tree_stump, x))
        
        # Produce a Boolean array indicating whether
        # each data point was correctly classified
        is_correct = predictions == target_values
        is_wrong   = predictions != target_values
        
        # Compute weighted error
        # YOUR CODE HERE
        weighted_error = (alpha * is_wrong).sum() / alpha.sum()
        
        # Compute model coefficient using weighted error
        # YOUR CODE HERE
        weight = .5 * log((1-weighted_error)/weighted_error)
        weights.append(weight)
        #print weight
        
        # Adjust weights on data point
        adjustment = is_correct.apply(lambda is_correct : exp(-weight) if is_correct else exp(weight))
        #print adjustment
        
        # Scale alpha by multiplying by adjustment 
        # Then normalize data points weights
        ## YOUR CODE HERE 
        alpha_sum = alpha.sum()
        alpha = alpha*adjustment / alpha_sum
        #print alpha
        
    return weights, tree_stumps


stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, features, target, num_tree_stumps=2)


def print_stump(tree):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    split_feature, split_value = split_name.split('.')
    print '                       root'
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]{1}[{0} == 1]    '.format(split_name, ' '*(27-len(split_name)))
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                 (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))


print_stump(tree_stumps[0])
print_stump(tree_stumps[1])
stump_weights

#Training a boosted ensemble of 10 stumps
stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, features, 
                                target, num_tree_stumps=10)

def predict_adaboost(stump_weights, tree_stumps, data):
    scores = sframe.SArray([0.]*len(data))
    
    for i, tree_stump in enumerate(tree_stumps):
        predictions = data.apply(lambda x: classify(tree_stump, x))
        
        # Accumulate predictions on scores array
        # YOUR CODE HERE
        scores = scores + (stump_weights * predictions)
        
    return scores.apply(lambda score : +1 if score > 0 else -1)

#Quiz Question 3: Are the weights monotonically decreasing, monotonically 
# increasing, or neither?
q3 = "neither"

predictions = predict_adaboost(stump_weights, tree_stumps, validation_data)
