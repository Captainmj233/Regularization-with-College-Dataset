import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt


college=pd.read_csv('data.csv',  dtype={'column': float})

college



college.describe()


# Encode categorical target variable as numeric values
le = LabelEncoder()
college['private'] = le.fit_transform(college['private'])



X = college[['s_f_ratio', 'enroll', 'perc_alumni']]
y =college['private']


#  ## RIDGE REGRESSION



from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV

X = college[['s_f_ratio', 'enroll', 'perc_alumni']]
y = college['private']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the range of alpha values to test
alphas = np.geomspace(0.0001,0.5, 10)

# Fit RidgeCV model to training data
ridge_cv = RidgeCV(alphas=alphas, cv=10)
ridge_cv.alphas_ = alphas  # Set alphas_ manually
ridge_cv.fit(X_train, y_train)

# Print lambda.min and lambda.1se
lambda_min = ridge_cv.alpha_
lambda_1se = lambda_min * np.exp(np.mean(np.log(ridge_cv.alphas_ / lambda_min)))
print("lambda.min:", lambda_min)
print("lambda.1se:", lambda_1se)


# The value of lambda.min is 0.5. This means that when the Ridge Regression model is trained with this value of lambda, it achieves the minimum mean squared error, which is the estimate of the prediction error rate.
# 
# The value of lambda.1se is 0.007071067811865479. This value of lambda is the largest value within one standard error of the minimum mean squared error. 



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Create StandardScaler object and transform the data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)

# Fit RidgeCV model to training data
ridge_cv = RidgeCV(alphas=alphas, cv=10)
ridge_cv.alphas_ = alphas  # Set alphas_ manually
ridge_cv.fit(X_train_std, y_train)

# Get cross-validation scores for each alpha
cv_scores = []
for alpha in ridge_cv.alphas_:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X_train_std, y_train, cv=10, scoring='neg_mean_squared_error')
    cv_scores.append(np.mean(np.abs(scores)))

# Plot the results
lambda_min = ridge_cv.alpha_
lambda_1se = np.min(ridge_cv.alphas_[np.where(cv_scores <= np.min(cv_scores) + 1)])
fig, ax = plt.subplots()
ax.plot(np.log10(ridge_cv.alphas_), cv_scores)
ax.axvline(np.log10(lambda_min), color='r', linestyle='--', label='lambda.min')
ax.axvline(np.log10(lambda_1se), color='g', linestyle='--', label='lambda.1se')
ax.set_xlabel('log(alpha)')
ax.set_ylabel('Cross-validation score')
ax.set_title('Ridge Regression')
ax.legend()
# Plot mean squared errors
mse = np.array(cv_scores) * -1
ax2 = ax.twinx()
ax2.plot(np.log10(ridge_cv.alphas_), mse, color='purple', linestyle='--', label='MSE')
ax2.set_ylabel('Mean Squared Error')
ax2.legend(loc='upper center')
fig.tight_layout()
plt.show()


# The graph demonstrates that at very low values of alpha, the mean squared error (MSE) initially starts high because the model is not severely punished and can overfit. The model becomes more regularized as alpha rises, and the MSE falls, reaching a minimum of about -1.5 on the log scale. Underfitting and a rise in the MSE result from further raising the alpha. The dotted lines show that a slightly greater value of alpha than lambda.min can still result in satisfactory performance. As a result, the selection of alpha is crucial, and the dashed lines can help in making this decision for the final model.



from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Create a Ridge regression object with a specified regularization parameter alpha
alpha = 1.0
ridge = Ridge(alpha=alpha)

# Create a StandardScaler object and transform the training data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)

# Fit the Ridge regression model to the training data
ridge.fit(X_train_std, y_train)

# Report on the coefficients
print("Intercept: ", ridge.intercept_)
for i in range(len(X_train.columns)):
    print(X_train.columns[i], ": ", ridge.coef_[i])


# The intercept value of 0.7274401473296501 represents the estimated mean value of the outcome variable when all predictor variables are equal to zero.
# 
# The coefficient for s_f_ratio is -0.1260606520442143. This means that for every one unit increase in s_f_ratio, the outcome variable is expected to decrease by -0.1260606520442143 units, while holding all other predictor variables constant.
# 
# The coefficient for enroll is -0.20450649485298666. This means that for every one unit increase in enroll, the outcome variable is expected to decrease by -0.20450649485298666 units, while holding all other predictor variables constant
# 
# The coefficient for perc_alumni is 0.09665903465623264. This means that for every one unit increase in perc_alumni, the outcome variable is expected to increase by 0.09665903465623264 units, while holding all other predictor variables constant


# Make predictions on the training set
y_train_pred = ridge.predict(X_train_std)

# Calculate RMSE
rmse = np.sqrt(np.mean((y_train - y_train_pred)**2))
print("RMSE on training set:", rmse)




# Scale the test data
X_test_std = scaler.transform(X_test)

# Predict the target variable for the test data
y_pred_test = ridge.predict(X_test_std)

# Calculate the root mean square error (RMSE) for the test data
rmse_test = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
print("RMSE for the test data:", rmse_test)

# Calculate the root mean square error (RMSE) for the training data
y_pred_train = ridge.predict(X_train_std)
rmse_train = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
print("RMSE for the training data:", rmse_train)

# Compare the RMSE for the training and test data
if rmse_test < rmse_train:
    print("The model is not overfit.")
else:
    print("The model is overfit.")


# RMSE for the test data is reported as 0.31944629418404874. This means that on average, the model's predictions are off by 0.31944629418404874 units of the outcome variable, when predicting on the test data. The lower the value of the RMSE, the better the model is at predicting the outcome variable. 
#  
# RMSE for the training data is reported as 0.3205503272742786. This means that on average, the model's predictions are off by 0.3205503272742786 units of the outcome variable, when predicting on the training data
#  
# This RMSE for the training data is slightly higher than the RMSE for the test data, which suggests that the model may be slightly overfitting to the training data.

# ## LASSO



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso


# Create LassoCV object
lasso_cv = LassoCV(cv=10, random_state=42)

# Fit model on training data
lasso_cv.fit(X_train, y_train)
# Extract lambda.min and lambda.1se
lambda_min = lasso_cv.alpha_
lambda_1se = np.max(lasso_cv.alphas_[(lasso_cv.mse_path_.mean(axis=1) + lasso_cv.mse_path_.std(axis=1)) < np.min(lasso_cv.mse_path_.mean(axis=1) + lasso_cv.mse_path_.std(axis=1)) + 0.01])

# Print lambda.min and lambda.1se
print("lambda.min:", lambda_min)
print("lambda.1se:", lambda_1se)


# The value of lambda.min is 0.22089273153376823. This means that when the LASSO model is trained with this value of lambda, it achieves the minimum mean cross-validated error, which is the estimate of the prediction error rate.
# 
# The value of lambda.1se is 0.38601632171053285. This value of lambda is the largest value within one standard error of the minimum cross-validated error.



# Plot mean squared error as a function of alpha
mse_mean = np.mean(lasso_cv.mse_path_, axis=1)
mse_std = np.std(lasso_cv.mse_path_, axis=1)
plt.plot(np.log10(lasso_cv.alphas_), mse_mean)
plt.fill_between(np.log10(lasso_cv.alphas_), mse_mean - mse_std, mse_mean + mse_std, alpha=0.2)
plt.axvline(np.log10(lambda_min), color='r', linestyle='--', label='lambda_min')
plt.axvline(np.log10(lambda_1se), color='g', linestyle='--', label='lambda_1se')

# Annotate lambda_min and lambda_1se on plot
plt.annotate(f'lambda_min = {lambda_min:.2e}', xy=(np.log10(lambda_min), mse_mean[lasso_cv.alphas_ == lambda_min]), 
             xytext=(np.log10(lambda_min)+0.5, mse_mean[lasso_cv.alphas_ == lambda_min]-0.05),
             arrowprops=dict(facecolor='red', shrink=0.1))
plt.annotate(f'lambda_1se = {lambda_1se:.2e}', xy=(np.log10(lambda_1se), mse_mean[lasso_cv.alphas_ == lambda_1se]), 
             xytext=(np.log10(lambda_1se)-0.5, mse_mean[lasso_cv.alphas_ == lambda_1se]-0.05),
             arrowprops=dict(facecolor='green', shrink=0.05))

# Label axes and add title
plt.xlabel('log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Lasso Regression: Mean Squared Error vs. Log of Alpha')

# Add grid lines
plt.grid()

# Add legend
plt.legend()

# Show plot
plt.show()


# The graph shows how the mean squared error shifts as the regularization parameter alpha is changed. The vertical dashed line for lambda.min, which corresponds to the least mean squared error, shows the ideal value of alpha. By examining the green line for lambda.1se, which is one standard deviation away from the lowest, we can also identify the range of alpha values that result in comparable performance. The graphic indicates that we are overfitting the data with several features because it is clear that raising alpha causes a rise in mean squared error. In order to create a simpler model, we should either select fewer features or utilize a higher value of alpha.



# Fit LASSO model with lambda.min
lasso = Lasso(alpha=lambda_min)
lasso.fit(X_train, y_train)

# Report on coefficients
coefficients = dict(zip(X.columns, lasso.coef_))
print("Coefficients:", coefficients)


# The coefficient for s_f_ratio is -0.016875240558383544, which means that for every one unit increase in s_f_ratio, the outcome variable is expected to decrease by -0.016875240558383544 units, holding all other variables constant. This suggests that s_f_ratio is negatively associated with the outcome variable, and higher values of s_f_ratio are expected to be associated with lower values of the outcome variable.
# 
# The coefficient for enroll is -0.00024603431060990423, which means that for every one unit increase in enroll, the outcome variable is expected to decrease by -0.00024603431060990423 units, holding all other variables constant. This suggests that enroll is also negatively associated with the outcome variable, but the effect is much smaller than the effect of s_f_ratio.
# 
# The coefficient for perc_alumni is 0.00820449195208882, which means that for every one unit increase in perc_alumni, the outcome variable is expected to increase by 0.00820449195208882 units, holding all other variables constant. This suggests that perc_alumni is positively associated with the outcome variable, and higher values of perc_alumni are expected to be associated with higher values of the outcome variable.



from sklearn.metrics import mean_squared_error

# Predict on training set
y_pred_train = lasso.predict(X_train)

# Calculate RMSE on training set
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

print("RMSE on training set:", rmse_train)


# RMSE is reported as 0.32543666167235963 on the training set. This means that on average, the model's predictions are off by 0.32543666167235963 units of the outcome variable, when predicting on the training set.


# Use the trained Lasso model to make predictions on the test set
y_pred = lasso.predict(X_test)

# Calculate the RMSE
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

print("RMSE on test set:", rmse)


# RMSE on the test set is reported as 0.3221382631189713. This means that on average, the model's predictions are off by 0.3221382631189713 units of the outcome variable, when predicting on the test set.

# ## Comparison

# Based on the RMSE values, the Ridge Regression model performed better than the Lasso Regression model, as it had a lower RMSE on both the training and test data. This suggests that the Ridge Regression model was more accurate in predicting the outcome variable and had better generalization performance on new, unseen data.
# 
# The lower RMSE on the test data in the Ridge Regression model suggests that the Ridge Regression model was able to better capture the underlying patterns in the data and make more accurate predictions on new data than the Lasso Regression model. The slightly lower RMSE on the training data in the Ridge Regression model also suggests that the Ridge Regression model was less overfit to the training data than the Lasso Regression model.
