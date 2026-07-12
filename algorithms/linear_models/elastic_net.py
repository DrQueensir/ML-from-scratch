import numpy as np

class ElasticNet:

    def __init__(self,learningrate=0.01,iterations=1000,lambda1=1.0,lambda2=1.0):
        self.weights=None
        self.bias=None

        self.learningrate=learningrate
        self.iterations=iterations
        self.lambda1=lambda1
        self.lambda2=lambda2

        self.cost_history=[]

    def fit(self,X,Y):
        samples,features=X.shape
        self.weights=np.zeros(features)
        self.bias=0.0

        for _ in range(self.iterations):
            y_pred=np.dot(X,self.weights)+self.bias
            y_error=y_pred-Y

            cost=((1/(2*samples))*np.sum(y_error**2))+((self.lambda1/(samples))*np.sum(np.abs(self.weights)))+((self.lambda2/(2*samples))*np.sum(self.weights**2))
            self.cost_history.append(cost)

            dw=((1/samples)*np.dot(X.T,y_error))+((self.lambda1/samples)*np.sign(self.weights))+((self.lambda2/samples)*self.weights)
            db=(1/samples)*np.sum(y_error)

            self.weights=self.weights-self.learningrate*dw
            self.bias=self.bias-self.learningrate*db

    def predict(self,X):
        return np.dot(X,self.weights)+self.bias

    def score(self, X, Y):
        predictions = self.predict(X)

        ss_res = np.sum((Y - predictions)**2)
        ss_tot = np.sum((Y - np.mean(Y))**2)

        return 1 - (ss_res / ss_tot)
