import sframe
loans = sframe.SFrame('lending-club-data.gl/')

safe_loan_label = +1
risky_loan_label = -1
target = 'safe_loans'

#reassign the labels to have +1 for a safe loan, and -1 for a risky (bad) loan.
loans[target] = loans['bad_loans'].apply(lambda x : safe_loan_label if x==0 else risky_loan_label)
loans = loans.remove_column('bad_loans')

features = ['grade',          # grade of the loan
            'term',           # the term of the loan
            'home_ownership', # home_ownership status: own, mortgage or rent
            'emp_length',     # number of years of employment
           ]

# Extract the feature columns and target column
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
#categorical_variables = []
#for feat_name, feat_type in zip(loans_data.column_names(), loans_data.column_types()):
#    if feat_type == str:
#        categorical_variables.append(feat_name)

for feature in features:
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
#Early stopping methods for decision trees
#-------------------------------------------

#Early stopping condition: Minimum node size
def reached_minimum_node_size(data, min_node_size):
    # Return True if the number of data points is less than or equal to the minimum node size.
    return len(data) <= min_node_size

#Quiz question 1: Given an intermediate node with 6 safe loans and 3 risky loans, 
# if the min_node_size parameter is 10, what should the tree learning algorithm do next?
q1 = "stop"

#Early stopping condition: Minimum gain in error reduction
def error_reduction(error_before_split, error_after_split):
    # Return the error before the split minus the error after the split.
    return error_before_split - error_after_split

#Quiz question 2: Assume an intermediate node has 6 safe loans and 3 risky loans. 
#For each of 4 possible features to split on, the error reduction is 
#0.0, 0.05, 0.1, and 0.14, respectively. If the minimum gain in error reduction 
#parameter is set to 0.2, what should the tree learning algorithm do next?
q2 = "stop"

#-------------------------------------------
#decision tree classifier implementation
#-------------------------------------------
'''
Computes the number of misclassified examples of an intermediate node given the 
set of labels (y values) of the data points contained in the node.
'''
def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0

    # Count the number of 1's (safe loans)
    safe_count = labels_in_node.apply(lambda x: 1 if x==safe_loan_label else 0).sum()
    # Count the number of -1's (risky loans)
    risky_count = labels_in_node.apply(lambda x: 1 if x==risky_loan_label else 0).sum()
        
    # Return the number of mistakes that the majority classifier makes.
    return safe_count if safe_count <= risky_count else risky_count

'''
The function will loop through the list of possible features, and consider 
splitting on each of them. It will calculate the classification error of each 
split and return the feature that had the smallest classification error when 
split on.
'''
def best_splitting_feature(data, features, target):
    
    best_feature = None # Keep track of the best feature 
    best_error = 10     # Keep track of the best error so far 
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))  
    
    # Loop through each feature to consider splitting on that feature
    for feature in features:
        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        
        # The right split will have all data points where the feature value is 1
        right_split =  data[data[feature] == 1]
            
        # Calculate the number of misclassified examples in the left split.
        left_mistakes = intermediate_node_num_mistakes(left_split[target])

        # Calculate the number of misclassified examples in the right split.
        right_mistakes = intermediate_node_num_mistakes(right_split[target])
            
        # Compute the classification error of this split.
        error = (left_mistakes + right_mistakes) / num_data_points

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
def create_leaf(target_values):    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True } 
   
    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == safe_loan_label])
    num_minus_ones = len(target_values[target_values == risky_loan_label])    

    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = safe_loan_label
    else:
        leaf['prediction'] = risky_loan_label

    # Return the leaf node
    return leaf

'''
Learns the decision tree recursively and implements 3 stopping conditions:

1. Stopping condition 1: All data points in a node are from the same class.
2. Stopping condition 2: No more features to split on.
3. Additional stopping condition3: In addition to the above two stopping we will 
   also consider a stopping condition based on 
   
   3a. the max_depth of the tree. 
   3b. the minimum node size.
   3c. the minimum error reduction.
   
   By not letting the tree grow beyond these thresholds, we will save 
   computational effort in the learning process and reduce overfitting. 
'''
def decision_tree_create(data, features, target, current_depth = 0, 
                        max_depth = 10, 
                        min_node_size = 1, 
                        min_error_reduction = 0.0):
                        
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))

    # Stopping condition 1
    # (Check if there are mistakes at current node.
    # Recall you wrote a function intermediate_node_num_mistakes to compute this.)
    if intermediate_node_num_mistakes(target_values) == 0:
        print "Reached 0 mistakes.  Stopping..."     
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)
    
    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if remaining_features == []:
        print "Reached 0 features.  Stopping.."    
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)    
    
    # Stopping condition 3a (limit tree depth)
    if current_depth >= max_depth:
        print "Reached maximum depth. Stopping for now."
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values)

    # Stopping condition 3b (minimum node size)
    if reached_minimum_node_size(data, min_node_size):
        print "Reached minimum node size. Stopping for now."
        return create_leaf(target_values)
    
    # Find the best splitting feature (recall the function best_splitting_feature implemented above)
    splitting_feature = best_splitting_feature(data, remaining_features, target)
    
    # Consider split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]

    # Stopping condition 3c (minimum error reduction)
    #determine classification error before any split
    before = intermediate_node_num_mistakes(target_values) / float(len(data))
    #determine classifcation error after split
    left_mistakes = intermediate_node_num_mistakes(left_split[target])
    right_mistakes = intermediate_node_num_mistakes(right_split[target])
    after = (left_mistakes + right_mistakes) / float(len(data))

#    diff = error_reduction(before, after)
#    shouldStop = diff <= min_error_reduction
#    print "Error before: {} Error after: {} Difference {} Should Stop {}".format(before, after, diff, shouldStop)
    
    if error_reduction(before, after) <= min_error_reduction:
        print "Minimum error reduction reached. Stopping for now."
        return create_leaf(target_values)    
    
    #no stopping conditions met, let's continue!
    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print "Creating leaf node."
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print "Creating leaf node."
        return create_leaf(right_split[target])
        
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth, min_node_size, min_error_reduction)        
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth, min_node_size, min_error_reduction)

    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}

#----------------------------------------
#train some trees
#----------------------------------------
my_decision_tree_new = decision_tree_create(train_data, features, target, 
                                            max_depth = 6,
                                            min_node_size = 100,
                                            min_error_reduction = 0.0)

my_decision_tree_old = decision_tree_create(train_data, features, target, 
                                            max_depth = 6,
                                            min_node_size = 0, 
                                            min_error_reduction=-1.0)

#----------------------------------------
#Making predictions with a decision tree
#----------------------------------------
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

print validation_data[0]
print 'Predicted class: %s ' % classify(my_decision_tree_new, validation_data[0], True)

classify(my_decision_tree_new, validation_data[0], annotate = True)
classify(my_decision_tree_old, validation_data[0], annotate = True)

#Quiz question 3: For my_decision_tree_new, is the prediction path for 
# validation_set[0] shorter, longer, or the same as for my_decision_tree_old
q3 = "shorter"

#Quiz question 4: For my_decision_tree_new, is the prediction path for any point 
# always shorter, always longer, always the same, shorter or the same, or 
# longer or the same as for my_decision_tree_old?
q4 = "shorter or the same"

#Quiz question 5: For a tree trained on any dataset using max_depth = 6, 
# min_node_size = 100, min_error_reduction=0.0, what is the maximum number of 
# splits encountered while making a single prediction?
q5 = 6

def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x))
    
    # Once you've made the predictions, calculate the classification error and return it
    return float((prediction != data[target]).sum()) / data.num_rows()

#Quiz question 6:  Is the validation error of the new decision tree 
# lower than, higher than, or the same as that of the old decision tree?
q6a = evaluate_classification_error(my_decision_tree_new, validation_data)
q6b = evaluate_classification_error(my_decision_tree_old, validation_data)
print "Validation data, classification error (new):", q6a
print "Validation data, classification error (old):", q6b
q6 = "slightly lower??"

#--------------------------------------------------
#Exploring the effect of max_depth
#--------------------------------------------------

model_1 = decision_tree_create(train_data, features, target, 
                               max_depth = 2,
                               min_node_size = 0,
                               min_error_reduction = -1.0)

model_2 = decision_tree_create(train_data, features, target, 
                               max_depth = 6,
                               min_node_size = 0,
                               min_error_reduction = -1.0)

model_3 = decision_tree_create(train_data, features, target, 
                               max_depth = 14,
                               min_node_size = 0,
                               min_error_reduction = -1.0)

print "Training data, classification error (model 1):", evaluate_classification_error(model_1, train_data)
print "Training data, classification error (model 2):", evaluate_classification_error(model_2, train_data)
print "Training data, classification error (model 3):", evaluate_classification_error(model_3, train_data)

print "Validation data, classification error (model 1):", evaluate_classification_error(model_1, validation_data)
print "Validation data, classification error (model 2):", evaluate_classification_error(model_2, validation_data)
print "Validation data, classification error (model 3):", evaluate_classification_error(model_3, validation_data)

#Quiz Question 7: Which tree has the smallest error on the validation data?
q7 = "model 3"

#Quiz Question 8: Does the tree with the smallest error in the training data 
# also have the smallest error in the validation data?
q8 = "yes"

#Quiz Question 9: Is it always true that the tree with the lowest classification 
# error on the training set will result in the lowest classification error in 
# the validation set?
q9 = "No"

#--------------------------------------------------
#Measuring the complexity of the tree
#--------------------------------------------------
'''
number of leaves in the tree
'''
def count_leaves(tree):
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])

#Quiz question 10: Which tree has the largest complexity?
model_1_count = count_leaves(model_1)
model_2_count = count_leaves(model_2)
model_3_count = count_leaves(model_3)

print "Leaves Count (model 1):", model_1_count
print "Leaves Count (model 2):", model_2_count
print "Leaves Count (model 3):", model_3_count

q10 = "model 3"

#Quiz question 11: Is it always true that the most complex tree will result in 
# the lowest classification error in the validation_set?
q11 = "no"

#--------------------------------------------------
#Exploring the effect of min_error
#--------------------------------------------------
model_4 = decision_tree_create(train_data, features, target, 
                               max_depth = 6,
                               min_node_size = 0,
                               min_error_reduction = -1.0)

model_5 = decision_tree_create(train_data, features, target, 
                               max_depth = 6,
                               min_node_size = 0,
                               min_error_reduction = 0)

model_6 = decision_tree_create(train_data, features, target, 
                               max_depth = 6,
                               min_node_size = 0,
                               min_error_reduction = 5)                               

print "Validation data, classification error (model 4):", evaluate_classification_error(model_4, validation_data)
print "Validation data, classification error (model 5):", evaluate_classification_error(model_5, validation_data)
print "Validation data, classification error (model 6):", evaluate_classification_error(model_6, validation_data)

model_4_count = count_leaves(model_4)
model_5_count = count_leaves(model_5)
model_6_count = count_leaves(model_6)

print "Leaves Count (model 4):", model_4_count
print "Leaves Count (model 5):", model_5_count
print "Leaves Count (model 6):", model_6_count

#Quiz Question 12: Using the complexity definition above, which model 
# (model_4, model_5, or model_6) has the largest complexity?
q12 = "model 4"

#Quiz Question 13: model_4 and model_5 have similar classification error on the 
# validation set but model_5 has lower complexity? Should you pick 
# model_5 over model_4?
q13 = "model_5, yes"

#--------------------------------------------------
#Exploring the effect of min_node_size
#--------------------------------------------------
model_7 = decision_tree_create(train_data, features, target, 
                               max_depth = 6,
                               min_node_size = 0,
                               min_error_reduction = -1.0)

model_8 = decision_tree_create(train_data, features, target, 
                               max_depth = 6,
                               min_node_size = 2000,
                               min_error_reduction = -1.0)

model_9 = decision_tree_create(train_data, features, target, 
                               max_depth = 6,
                               min_node_size = 50000,
                               min_error_reduction = -1.0)

print "Validation data, classification accuracy (model 7):", (1.0 - evaluate_classification_error(model_7, validation_data))
print "Validation data, classification accuracy (model 8):", (1.0 - evaluate_classification_error(model_8, validation_data))
print "Validation data, classification accuracy (model 9):", (1.0 - evaluate_classification_error(model_9, validation_data))

model_7_count = count_leaves(model_7)
model_8_count = count_leaves(model_8)
model_9_count = count_leaves(model_9)

print "Leaves Count (model 7):", model_7_count
print "Leaves Count (model 8):", model_8_count
print "Leaves Count (model 9):", model_9_count

#Quiz Question 14: Using the results obtained in this section, which model 
# (model_7, model_8, or model_9) would you choose to use?
q14 = "model 8"


















