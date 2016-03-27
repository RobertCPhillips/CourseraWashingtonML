import sys
print(sys.version)

import sframe
loans = sframe.SFrame('lending-club-data.gl/')
loans.column_names()

# safe_loans =  1 => safe
# safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')

print 'The proportion of safe loans is ', loans['safe_loans'].apply(lambda x : +1 if x==1 else 0).mean()

#the features and target that we want to use
features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]

#-------------------------------------------
#Sample data to balance classes
#-------------------------------------------
safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]

print "Number of safe loans  : %s" % len(safe_loans_raw)
print "Number of risky loans : %s" % len(risky_loans_raw)

# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))

risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(percentage, seed=1)

# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans)
print 'The proportion of safe loans is ', loans_data['safe_loans'].apply(lambda x : +1 if x==1 else 0).mean()


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

#-------------------------------------------
#Build a decision tree classifier
#-------------------------------------------

#Split data into training and validation
train_data, validation_data = loans_data.random_split(.8, seed=1)

'''
input parameters:
  data_sframe: a data frame to be converted
  label: a string, containing the name of the single column that is used as class labels.

output:
  a 2D array for features
  a 1D array for class labels
'''
def get_numpy_data(data_sframe, label):
    label_sarray = data_sframe[label]
    label_array = label_sarray.to_numpy()
    
    data_sframe.remove_column(label)
    feature_matrix = data_sframe.to_numpy()
    
    return(feature_matrix, label_array)
    
feature_matrix_train, label_train = get_numpy_data(train_data, target)

from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier(max_depth=6)
decision_tree_model.fit(feature_matrix_train, label_train)

small_model = DecisionTreeClassifier(max_depth=2)
small_model.fit(feature_matrix_train, label_train)

#visualize
from sklearn import tree
from StringIO import StringIO 
from IPython.display import Image
out = StringIO()
tree.export_graphviz(small_model, out_file=out,
                     feature_names=train_data.column_names(),  
                     class_names=['risky', 'safe'],  
                     filled=True, rounded=True)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())


#-------------------------------------------
#Making predictions
#-------------------------------------------

#grab 2 positive examples and 2 negative examples
validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_data

#predict class on the samples
sample_validation_data_matrix, sample_validation_data_label = get_numpy_data(sample_validation_data, target)
sample_validation_data_predict = decision_tree_model.predict(sample_validation_data_matrix)

#question 1: What percentage of the predictions on sample_validation_data did decision_tree_model get correct?
q1 = decision_tree_model.score(sample_validation_data_matrix, sample_validation_data_label)

#get prediction probability
sample_validation_data_predictproba = decision_tree_model.predict_proba(sample_validation_data_matrix)

#get prediction probability
sample_validation_data_predictproba2 = small_model.predict_proba(sample_validation_data_matrix)

#question 2: Notice that the probability preditions are the exact same for the 2nd and 3rd loans. Why would this happen?
q2 = "they follow the same path"

sample_validation_data[1]
#question 3: Based on the visualized tree, what prediction would you make for this data point (according to small_model)?
q3 = "+1"

#-----------------------------------------------------
#Evaluating accuracy of the decision tree models
#-----------------------------------------------------

#train accuracy
decision_tree_model_train_score = decision_tree_model.score(feature_matrix_train, label_train)
small_model_train_score = small_model.score(feature_matrix_train, label_train)

#validation accuracy
validation_data_matrix, validation_data_label = get_numpy_data(validation_data, target)

decision_tree_model_valid_score = decision_tree_model.score(validation_data_matrix, validation_data_label)
small_model_valid_score = small_model.score(validation_data_matrix, validation_data_label)

#question 4: What is the accuracy of decision_tree_model on the validation set, rounded to the nearest .01?
q4 = round(decision_tree_model_valid_score, 2)


#-----------------------------------------------------
#Evaluating accuracy of a complex decision tree model
#-----------------------------------------------------
big_model = DecisionTreeClassifier(max_depth=10)
big_model.fit(feature_matrix_train, label_train)

big_model_train_score = big_model.score(feature_matrix_train, label_train)
big_model_valid_score = big_model.score(validation_data_matrix, validation_data_label)

#question 5: How does the performance of big_model on the validation set compare to decision_tree_model on the validation set? Is this a sign of overfitting?
q5 = "A little better.  Maybe, it only went up a little, but valid performance also went down a little."

#-----------------------------------------------------
#Quantifying the cost of mistakes
#-----------------------------------------------------

#predict validation data
decision_tree_model_valid_predict = decision_tree_model.predict(validation_data_matrix)

from sklearn.metrics import confusion_matrix
cf = confusion_matrix(validation_data_label, decision_tree_model_valid_predict)

#question 6: Quiz Question: Let's assume that each mistake costs us money: 
#  a false negative costs $10,000, while a false positive positive costs $20,000. 
#  What is the total cost of mistakes made by decision_tree_model on validation_data?
q6 = cf[0,1] * 20000. + cf[1,0] * 10000.

fp = 0
fn = 0

for i in xrange(len(decision_tree_model_valid_predict)):
    if validation_data_label[i] == 1:
        if decision_tree_model_valid_predict[i] == -1:
            fn = fn + 1
    else:
        if decision_tree_model_valid_predict[i] == 1:
            fp = fp + 1
