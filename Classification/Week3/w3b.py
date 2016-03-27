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

features = loans_data.column_names()
features.remove(target)  # Remove the response variable

#-------------------------------------------
#Split data into training and validation
#-------------------------------------------
train_data, validation_data = loans_data.random_split(.8, seed=1)

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
3. Additional stopping condition: In addition to the above two stopping we will 
   also consider a stopping condition based on the max_depth of the tree. 
   By not letting the tree grow too deep, we will save computational effort in 
   the learning process.    
'''
def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))

    # Stopping condition 1
    # (Check if there are mistakes at current node.
    # Recall you wrote a function intermediate_node_num_mistakes to compute this.)
    if intermediate_node_num_mistakes(target_values) == 0:
        print "Stopping condition 1 reached."     
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)
    
    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if remaining_features == []:
        print "Stopping condition 2 reached."    
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)    
    
    # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth:
        print "Reached maximum depth. Stopping for now."
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values)

    # Find the best splitting feature (recall the function best_splitting_feature implemented above)
    splitting_feature = best_splitting_feature(data, remaining_features, target)
    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
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
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth)        
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth)

    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}

#----------------------------------------
#train a tree
#----------------------------------------
my_decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6)

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

validation_data[0]
print 'Predicted class: %s ' % classify(my_decision_tree, validation_data[0], True)


#question 1: What was the feature that my_decision_tree first split on while making the prediction for test_data[0]?
q1 = 'term. 36 months'

#question 2: What was the first feature that lead to a right split of test_data[0]?
q2 = 'grade.D'

#question 3: What was the last feature split on before reaching a leaf node for test_data[0]?
q3 = 'grade.D'

def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x))
    
    # Once you've made the predictions, calculate the classification error and return it
    return float((prediction != data[target]).sum()) / data.num_rows()

#question 4: What is the classification error of my_decision_tree on the validation_data?
q4 = evaluate_classification_error(my_decision_tree, validation_data)


#--------------------------------------------------
#Printing out a decision stump
#--------------------------------------------------

def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    split_feature, split_value = split_name.split('.')
    print '                       %s' % name
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))

print_stump(my_decision_tree)

#question 5: What is the feature that is used for the split at the root node?
q5 = 'term. 36 months'

print_stump(my_decision_tree['left'], my_decision_tree['splitting_feature'])
print_stump(my_decision_tree['left']['left'], my_decision_tree['left']['splitting_feature'])


#question 6: What is the path of the first 3 feature splits considered along the left-most branch of my_decision_tree?
q6 = '{} --> {} --> {} --> {}'.format(my_decision_tree['splitting_feature'], \
                                       my_decision_tree['left']['splitting_feature'], \
                                       my_decision_tree['left']['left']['splitting_feature'], \
                                       my_decision_tree['left']['left']['left']['splitting_feature'])

#question 7: What is the path of the first 3 feature splits considered along the right-most branch of my_decision_tree?
q7 = '{} --> {} --> {}'.format(my_decision_tree['splitting_feature'], \
                                       my_decision_tree['right']['splitting_feature'], \
                                       my_decision_tree['right']['right']['splitting_feature'])


















