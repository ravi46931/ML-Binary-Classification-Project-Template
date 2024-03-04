import numpy as np
import termcolor
from src.logger import logging
from src.constants import *

class LogisticRegression:
    def __init__(self):
        self._weight=None
#         self._bias=None
        self._cost=None
    
    @property
    def weight(self):
        return self._weight
    
#     @property
#     def bias(self):
#         return self._bias
    
    @property
    def cost(self):
        return self._cost
    
    def logistic(self, x, w):
    
         # Dot product of input x and weight vector w
        dot_product=np.dot(x,w)

         # Exponent term using the logistic function formula
        exponent=1.0/(1+np.exp(-dot_product))

        return exponent
    
    def calculate_cost (self, X, y, W):
        # Dimensions of the input matrix X
        m,n=X.shape

        loss=0
        # Iterate through all the data points
        for i in range(m):

            # Accumulate the loss
            loss=loss + (y[i]*np.log(self.logistic(X[i],W))+(1-y[i])*np.log(1-self.logistic(X[i],W)))

        # Return the cost 
        return (-1.0/m)*loss
    
    def calculate_accuracy(self, X, y, W):
    
        # Variable for true prediction
        count_true=0

        # Iterate through each data point
        for i in range(len(y)):

            if  ((self.logistic(W,X[i])>THRESOLD_VALUE) and (y[i]==1)) or ((self.logistic(W,X[i])<=THRESOLD_VALUE) and (y[i]==0)) :
                count_true += 1

        # return the accuracy in terms of percentage
        return (count_true)*100/len(y)

    def calculate_gradient(self, X, y, W):
        sum_gradients=0

        # Dimensions of the input matrix X
        m,n=X.shape

        # Iterate through each data point
        for i in range(m):

            # Difference between Actual and Predicted label
            diff=self.logistic(X[i],W) - y[i]

             # Accumulate the gradient
            sum_gradients=sum_gradients+diff*X[i]

        return sum_gradients 

    def gradient_descent(self, X, y, iterations, alpha):
        # number of rows (data points), number of columns of the data
        m,n=X.shape
        # Weight initialization (including the bias)
        W=np.random.random_sample(n)
        # Store the cost at each iterations
        cost=np.zeros(iterations)

        # Updating the weights 
        for i in range(iterations):

            # Update weights and bias
            W=W-(alpha/m)*self.calculate_gradient(X, y, W)

            # Cost 
            cost[i]=self.calculate_cost(X, y, W)

            # Model's accuracy
            accuracy=self.calculate_accuracy(X, y, W)

            # Prints informations of cost and accuray at each iterations
            if i == iterations-1:
                termcolor.cprint(f"Iterations: {i}, cost: {cost[i]:.4f}, Accuracy: {accuracy:.2f}%",'red', attrs=['bold'])
                logging.info(f"Iterations: {i}, cost: {cost[i]:.4f}, Accuracy: {accuracy:.2f}%")
            else:
                print(f"Iterations: {i}, cost: {cost[i]:.4f}, Accuracy: {accuracy:.2f}%")

        self._weight=W
        self._cost=cost

        
    def fit(self, X,y, iterations=ITERATIONS, alpha=ALPHA):
        self.gradient_descent(X, y, iterations, alpha)
    
    def predict(self, X):
        pred=[]
        for i in range(X.shape[0]):
            if (self.logistic(self._weight,X[i]))>THRESOLD_VALUE: 
                pred.append(1)
            else:
                pred.append(0)
        return pred
    