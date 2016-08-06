import sframe
sales = sframe.SFrame('kc_house_data.gl/')

train_data, test_data = sales.random_split(.8,seed=0)

'''
Accepts a column of data (e.g, an SArray) 'input_feature' and another column 
'output' and returns the Simple Linear Regression parameters 'intercept' and 
'slope'. Use the closed form solution to calculate the slope and intercept.
'''
def simple_linear_regression(input_feature, output):
    #The number of data points (N)
    N = len(input_feature)

    #The sum (or mean) of the Ys
    mean_y = 0.
    #The sum (or mean) of the Xs
    mean_x = 0.
    #The sum (or mean) of the product of the Xs and the Ys
    mean_xy = 0.
    #The sum (or mean) of the Xs squared
    mean_xx = 0.
    for i in range(N):
        mean_y += output[i]
        mean_x += input_feature[i]
        mean_xy += input_feature[i] * output[i]
        mean_xx += input_feature[i]**2
    
    mean_y = mean_y / N
    mean_x = mean_x / N
    mean_xy = mean_xy / N
    mean_xx = mean_xx / N
    
    numerator = mean_xy - mean_x*mean_y
    denominator = mean_xx - mean_x*mean_x
    
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    
    return(intercept, slope)

#Use your function to calculate the estimated slope and intercept on the 
#training data to predict price given sqft_living.
input_feature = train_data['sqft_living']
output = train_data['price']

intercept, slope = simple_linear_regression(input_feature, output)

'''
accepts a column of data 'input_feature', the 'slope', and the 'intercept' you 
learned, and returns an a column of predictions 'predicted_output' for each 
entry in the input column.
'''
def get_regression_predictions(input_feature, intercept, slope):
    predicted_output = input_feature.apply(lambda x: intercept + x*slope)
    return(predicted_output)

'''
accepts column of data: 'input_feature', and 'output' and the regression 
parameters 'slope' and 'intercept' and outputs the Residual Sum of Squares
'''
def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    predicted_output = get_regression_predictions(input_feature, intercept, slope)
    rss = 0
    for i in range(len(predicted_output)):
        rss += (output[i] - predicted_output[i])**2
    
    return(rss)


# Quiz Question 1: Using your Slope and Intercept from (4), What is the predicted 
# price for a house with 2650 sqft?
q1 = intercept + slope*2650

# Quiz Question 2: According to this function and the slope and intercept from 
# (4) What is the RSS for the simple linear regression using squarefeet to 
# predict prices on TRAINING data?
q2 = get_residual_sum_of_squares(input_feature, output, intercept, slope)

'''
Accept a column of data:'output' and the regression parameters 'slope' and 
'intercept' and outputs the column of data: 'estimated_input'.
'''
def inverse_regression_predictions(output, intercept, slope):
    #[your code here]
    return(estimated_input)

# Quiz Question 3: According to this function and the regression slope and 
# intercept from (3) what is the estimated square-feet for a house costing 
# $800,000?
q3 = inverse_regression_predictions(800000., intercept, slope)

#calculate the Simple Linear Regression slope and intercept for estimating price 
#based on bedrooms
input_feature_bedroom = train_data['bedrooms']

intercept_bedroom, slope_bedroom = simple_linear_regression(input_feature_bedroom, output)

# Quiz Question 4: Which model (square feet or bedrooms) has lowest RSS on TEST 
# data?
rss_bedroom = get_residual_sum_of_squares(input_feature_bedroom, output, intercept_bedroom, slope_bedroom)
q4 = "square feet" if q2 < rss_bedroom else "bedroom"

