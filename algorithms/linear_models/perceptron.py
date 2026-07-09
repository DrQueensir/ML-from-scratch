import numpy as np 

class Perceptron:

    def __init__(self,eta=0.01,iterations=1000):
        self.weights=None
        self.bias=None
        self.errors=[]

        self.eta=eta
        self.iterations=iterations

    def fit(self,X,Y):
        samples,features=X.shape
        self.weights=np.zeros(features)
        self.bias=0
        self.errors=[]

        for _ in range(self.iterations):
            indices=np.random.permutation(samples)
            for i in indices:
                x_i=X[i]
                y_i=Y[i]

                linear= np.dot(x_i,self.weights)+ self.bias
                y_pred=1 if linear >=0 else 0
                y_error=y_i-y_pred
                self.errors.append(y_error)

                self.weights=self.weights+self.eta*y_error*x_i
                self.bias=self.bias+self.eta*y_error

    def predict(self,X):
        linear=np.dot(X,self.weights)+self.bias
        return (linear>=0).astype(int)
    
    def score(self,X,Y):
        predictions = self.predict(X)
        return np.mean(predictions == Y)