# ML Project - Supervised Learning

![image](https://github.com/ssabiescu/asd/assets/156011844/3790fd94-4167-4e0b-bab3-38769002c816)

## 1. **Linear Regression**
  ### Description
##### 🏘️ This script receives as a training set information such as area, number of bedrooms, etc. and price about several houses in California.
##### 🏠 Based on this data, the model is trained and uses linear regression and is able to estimate the price of a new house.


### The key steps in the script include:
### 
- Loading the house data from a text file.
- Normalizing the data using Z-score normalization.
- Fitting a Linear Regression model using Stochastic Gradient Descent.
- Making predictions using the trained model.
- Plotting the predictions against the original features to visualize the model's performance.

##### a) The normal Linear Regression
![image](https://github.com/ssabiescu/asd/assets/156011844/78f91b2d-d6ed-41ea-8c22-b1c2763d8bb2)

This visualization presents a comparison between actual and predicted housing prices from a linear regression model. The plots demonstrate the model’s predictions (in red) 🔴 against the actual values (in blue) 🔵

##### b) The multiple variable Linear Regression
![image](https://github.com/ssabiescu/asd/assets/156011844/720b9ac5-61db-4959-84e0-753e471f707d)

These graphs show the decreasing cost over iterations for a multiple variable linear regression model, indicating optimization of the model's parameters.
The detailed view on the right highlights the tail end of this process, where the cost reduction slows as the model approaches its best fit.

## 2. **Logistic Regression**
   ### Description
##### 🏫 This logistic regression model predict whether a student gets admitted into a university.
##### 🗒️ Admission is based on two tests and the training set contains the old results of several applicants and whether they were accepted or rejected.

### Essential steps in the script:

- Loading historical exam and admission data.
- Applying logistic regression to estimate admission probabilities.
- Optimizing the model with gradient descent.
- Making predictions and evaluating the model's accuracy.

![image](https://github.com/ssabiescu/asd/assets/156011844/422582de-f7f0-4498-84b6-ca15ea5d2c29)

With this model, you can then predict if a new student will be admitted based on their scores on the two exams.










