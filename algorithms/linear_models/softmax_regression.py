import numpy as np

class SoftmaxRegression:

    def __init__(self,learningrate=0.01,iterations=1000):
        self.weights=None
        self.bias=None

        self.learningrate=learningrate
        self.iterations=iterations

        self.cost_history=[]

    def _softmax(self,logits):
        exponent=np.exp(logits)
        total=np.sum(exponent,axis=1,keepdims=True)
        return exponent/total

    def _one_hot(self,Y):
        classes=np.max(Y)+1
        one_hot=np.zeros((len(Y),classes))
        for i in range(len(Y)):
            one_hot[i][Y[i]]=1
        return one_hot
    
    def fit(self,X,Y):
        samples,features=X.shape
        classes=len(np.unique(Y))

        self.weights=np.zeros((features,classes))
        self.bias=np.zeros(classes)
        for _ in range(self.iterations):
            logits=np.dot(X,self.weights)+self.bias
            probabilities=self._softmax(logits)
            Y_one_hot=self._one_hot(Y)
            cost=-np.mean(np.sum(Y_one_hot*np.log(probabilities+ 1e-15),axis=1))
            self.cost_history.append(cost)

            error=probabilities-Y_one_hot
            dw=(1/samples)*np.dot(X.T,error)
            db=(1/samples)*np.sum(error,axis=0)

            self.weights=self.weights-self.learningrate*dw
            self.bias=self.bias-self.learningrate*db


    def predict(self,X):
        logits=np.dot(X,self.weights)+self.bias
        probabilities=self._softmax(logits)
        return np.argmax(probabilities,axis=1)

    def score(self,X,Y):
        predictions=self.predict(X)
        return np.mean(predictions==Y)
