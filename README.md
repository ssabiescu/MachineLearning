# ML Project - Supervised Learning

![ai](https://github.com/ssabiescu/MachineLearning/assets/156011844/e6312803-c154-4266-9830-b3b05092343b)

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
![lin_reg](https://github.com/ssabiescu/MachineLearning/assets/156011844/ab371d58-72bd-4b49-886c-b3634b809ce9)


This visualization presents a comparison between actual and predicted housing prices from a linear regression model. The plots demonstrate the model’s predictions (in red) 🔴 against the actual values (in blue) 🔵

##### b) The multiple variable Linear Regression
![lin_reg_mul](https://github.com/ssabiescu/MachineLearning/assets/156011844/8ab032d9-8863-4c36-a07d-042793404bff)


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

![university](https://github.com/ssabiescu/MachineLearning/assets/156011844/870d0a9c-2111-45b0-84af-2030b472972c)

With this model, you can then predict if a new student will be admitted based on their scores on the two exams.










