import numpy as np
from linear_regression import LinearRegression

class PolynomialRegression:
    
    def __init__(self,learningrate=0.01,iterations=1000,degree=2):
        self.degree=degree
        self.model=LinearRegression(learningrate,iterations)

    def _transform(self,X):
        features=[]
        for d in range(1,self.degree+1):
            features.append(X**d)
        return np.hstack(features)

    def fit(self,X,Y):
        X_poly=self._transform(X)
        self.model.fit(X_poly,Y)

    def predict(self,X):
        X_poly=self._transform(X)
        self.model.predict(X_poly)

    def score(self,X,Y):
        predictions = self.predict(X)
        return np.mean(predictions == Y)