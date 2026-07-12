import numpy as np 


class LogisticRegression:

    def __init__(self,learningrate=0.01,iterations=1000):

        self.weights=None
        self.bias=None

        self.learningrate=learningrate
        self.iterations=iterations

        self.cost_history=[]

    def _sigmoid(self,Z):
        return 1/ (1 + np.exp(-Z))
    
    def _predict_probability(self,X):
        linear = np.dot(X,self.weights)+self.bias

        return self._sigmoid(linear)

    def fit(self,X,Y):
        samples,features=X.shape

        self.weights=np.zeros(features)
        self.bias=0.0

        for _ in range(self.iterations):
            y_pred=self._predict_probability(X)
            y_error=y_pred-Y

            cost=-(1/samples)*np.sum(Y*np.log(y_pred)+(1-Y)*np.log(1-y_pred))
            self.cost_history.append(cost)

            dw=(1/samples)*np.dot(X.T,y_error)
            db=(1/samples)*np.sum(y_error)

            self.weights=self.weights-self.learningrate*dw
            self.bias=self.bias-self.learningrate*db
            

    def predict(self,X):
        probability= self._predict_probability(X)
        return (probability>=0.5).astype(int)
    
    def score(self,X,Y):
        predictions = self.predict(X)
        return np.mean(predictions == Y)
        
