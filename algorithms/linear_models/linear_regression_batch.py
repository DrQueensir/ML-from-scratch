import numpy as np 

class LinearRegression:
    def __init__(self, learningrate=0.01,iterations=1000):
        self.weights=None
        self.bias=None

        self.learningrate=learningrate
        self.iterations=iterations

        self.cost_history=[]

    def fit(self,X,Y):
        samples,features=X.shape
        self.weights=np.zeros(features)
        self.bias=0

        for _ in range(self.iterations):
            y_pred=np.dot(X,self.weights)+self.bias
            y_error=y_pred-Y
            cost = (1 / (2 * samples)) * np.sum(y_error ** 2)
            self.cost_history.append(cost)

            dw=(1/samples)*np.dot(X.T,y_error)
            db=(1/samples)*np.sum(y_error)

            self.weights=self.weights - (self.learningrate*dw)
            self.bias=self.bias - (self.learningrate*db)




    def predict(self,X):
        return np.dot(X,self.weights)+self.bias