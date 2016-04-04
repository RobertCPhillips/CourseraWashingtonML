import sframe
loans = sframe.SFrame('lending-club-data.gl/')

safe_loan_label = +1
risky_loan_label = -1
target = 'safe_loans'

#reassign the labels to have +1 for a safe loan, and -1 for a risky (bad) loan.
loans[target] = loans['bad_loans'].apply(lambda x : safe_loan_label if x==0 else risky_loan_label)
loans = loans.remove_column('bad_loans')

features = ['grade',                     # grade of the loan (categorical)
            'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'payment_inc_ratio',         # ratio of the monthly payment to income
            'delinq_2yrs',               # number of delinquincies
             'delinq_2yrs_zero',         # no delinquincies in last 2 years
            'inq_last_6mths',            # number of creditor inquiries in last 6 months
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'open_acc',                  # number of open credit accounts
            'pub_rec',                   # number of derogatory public records
            'pub_rec_zero',              # no derogatory public records
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            'int_rate',                  # interest rate of the loan
            'total_rec_int',             # interest received to date
            'annual_inc',                # annual income of borrower
            'funded_amnt',               # amount committed to the loan
            'funded_amnt_inv',           # amount committed by investors for the loan
            'installment',               # monthly payment owed by the borrower
           ]

#-------------------------------------------------------------------------
#Skipping observations with missing values
#-------------------------------------------------------------------------
loans, loans_with_na = loans[[target] + features].dropna_split()

# Count the number of rows with missing data
num_rows_with_na = loans_with_na.num_rows()
num_rows = loans.num_rows()
print 'Dropping %s observations; keeping %s ' % (num_rows_with_na, num_rows)

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
#Gradient boosted tree classifier
#-------------------------------------------

def get_numpy_data(data_sframe, features, label):
    features_sframe = data_sframe[features]
    feature_matrix = features_sframe.to_numpy()
    
    label_sarray = data_sframe[label]
    label_array = label_sarray.to_numpy()

    return(feature_matrix, label_array)

train_matrix, train_label = get_numpy_data(train_data, features, target)

from sklearn import ensemble 
model_5 = ensemble.GradientBoostingClassifier(max_depth=6, n_estimators=5)
model_5.fit(train_matrix, train_label)

#-------------------------------------------
#Making predictions
#-------------------------------------------

#grab 2 positive examples and 2 negative examples
validation_safe_loans = validation_data[validation_data[target] == safe_loan_label]
validation_risky_loans = validation_data[validation_data[target] == risky_loan_label]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_data

sample_validation_data_predict = model_5.predict(sample_validation_data[features].to_numpy())
sample_validation_data[target]

#Quiz question 1: What percentage of the predictions on sample_validation_data 
# did model_5 get correct?
score = 0.0

for i in xrange(len(sample_validation_data[target])):
    if sample_validation_data[target][i] == sample_validation_data_predict[i]:
        score = score + 1.

q1 = score / len(sample_validation_data[target]) 

#Quiz question 2: Which loan (in sample_validation_data) has the highest 
# probability of being classified as a safe loan?
sample_validation_data_predict_prob = model_5.predict_proba(sample_validation_data[features].to_numpy())
q2 = "the 4th one"

#-------------------------------------------
#Evaluating the model on the validation data
#-------------------------------------------
validation_matrix, valdation_label = get_numpy_data(validation_data, features, target)

model_5_validation_accuracy = model_5.score(validation_matrix, valdation_label)
model_5_validation_predict = model_5.predict(validation_matrix)

false_pos_count = 0
false_neg_count = 0

for i in xrange(len(model_5_validation_predict)):
    if model_5_validation_predict[i] == safe_loan_label:
        if valdation_label[i] == risky_loan_label:
            false_pos_count += 1
    else:
         if valdation_label[i] == safe_loan_label:
             false_neg_count += 1

#Quiz question 3: What is the number of false positives on the validation_data?
q3 = false_pos_count

#-------------------------------------------
#Comparison with decision trees
#-------------------------------------------
model_5_cost = 10000. * false_neg_count  + 20000. * false_pos_count
q4 = model_5_cost

#-------------------------------------------
#Most positive & negative loans
#-------------------------------------------
model_5_validation_predict_prob = model_5.predict_proba(validation_matrix)

validation_data['proba_pos'] = model_5_validation_predict_prob[:,1]
validation_data['proba_neg'] = model_5_validation_predict_prob[:,0]

grades = ['grade.A',
          'grade.B',
          'grade.C',
          'grade.D',
          'grade.E',
          'grade.F',
          'grade.G']

#Quiz question 5: What grades are the top 5 loans?          
q5a = validation_data.topk('proba_pos', k=5)[grades]
q5b = validation_data.topk('proba_neg', k=5, reverse=True)[grades]


#-------------------------------------------
#Effects of adding more trees
#-------------------------------------------
model_10 = ensemble.GradientBoostingClassifier(max_depth=6, n_estimators=10)
model_10.fit(train_matrix, train_label)

model_50 = ensemble.GradientBoostingClassifier(max_depth=6, n_estimators=50)
model_50.fit(train_matrix, train_label)

model_100 = ensemble.GradientBoostingClassifier(max_depth=6, n_estimators=100)
model_100.fit(train_matrix, train_label)

model_200 = ensemble.GradientBoostingClassifier(max_depth=6, n_estimators=200)
model_200.fit(train_matrix, train_label)

model_500 = ensemble.GradientBoostingClassifier(max_depth=6, n_estimators=500)
model_500.fit(train_matrix, train_label)

#Compare accuracy on entire validation set
model_10_validation_accuracy = model_10.score(validation_matrix, valdation_label)
model_50_validation_accuracy = model_50.score(validation_matrix, valdation_label)
model_100_validation_accuracy = model_100.score(validation_matrix, valdation_label)
model_200_validation_accuracy = model_200.score(validation_matrix, valdation_label)
model_500_validation_accuracy = model_500.score(validation_matrix, valdation_label)

#Quiz Question 6: Which model has the best accuracy on the validation_data?
q6 = "model 200"

#Quiz Question 7: Is it always true that the model with the most trees will perform best on test data?
q7 = "no"


#Plot the training and validation error vs. number of trees
import matplotlib.pyplot as plt

def make_figure(dim, title, xlabel, ylabel, legend):
    plt.rcParams['figure.figsize'] = dim
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(loc=legend, prop={'size':15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

model_10_train_err = 1. - model_10.score(train_matrix, train_label)
model_50_train_err = 1. - model_50.score(train_matrix, train_label)
model_100_train_err = 1. - model_100.score(train_matrix, train_label)
model_200_train_err = 1. - model_200.score(train_matrix, train_label)
model_500_train_err = 1. - model_500.score(train_matrix, train_label)

training_errors = [model_10_train_err,
                   model_50_train_err,
                   model_100_train_err,
                   model_200_train_err,
                   model_500_train_err]

validation_errors = [1. - model_10_validation_accuracy,
                     1. - model_50_validation_accuracy,
                     1. - model_100_validation_accuracy,
                     1. - model_200_validation_accuracy,
                     1. - model_500_validation_accuracy]

plt.plot([10, 50, 100, 200, 500], training_errors, linewidth=4.0, label='Training error')
plt.plot([10, 50, 100, 200, 500], validation_errors, linewidth=4.0, label='Validation error')

make_figure(dim=(10,5), title='Error vs number of trees',
            xlabel='Number of trees',
            ylabel='Classification error',
            legend='best')


#Quiz question: Does the training error reduce as the number of trees increases?
q8 = "yes"

#Quiz question: Is it always true that the validation error will reduce as the number of trees increases?
q9 = "no"
