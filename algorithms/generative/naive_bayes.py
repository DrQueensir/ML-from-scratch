import numpy as np 

class NaiveBayes:

    def __init__(self):
        self.classes=None
        self.means=None
        self.variances=None
        self.priors=None

    def _gaussian(self,x, mean,variance):
        variance = variance + 1e-15
        exponent= np.exp(-((x-mean)**2)/(2*variance))
        coefficient=1/np.sqrt(2*np.pi*variance)
        return exponent*coefficient

    def fit(self,X,Y):
        samples,features= X.shape
        self.classes=np.unique(Y)
        num_classes=len(self.classes)
        self.means=np.zeros((num_classes,features))
        self.variances=np.zeros((num_classes,features))
        self.priors=np.zeros(num_classes)

        for index,c in enumerate(self.classes):
            X_class=X[Y==c]
            self.means[index]=np.mean(X_class,axis=0)
            self.variances[index]=np.var(X_class,axis=0)
            self.priors[index]=len(X_class)/samples
    
    def _predict(self,x):
        posteriors=[]
        for index,c in enumerate(self.classes):
            prior=np.log(self.priors[index])
            likelihood=self._gaussian(x,self.means[index],self.variances[index])
            likelihood = np.sum(np.log(likelihood))
            posteriors.append(prior+likelihood)
        return self.classes[np.argmax(posteriors)]

    def predict(self,X):
        predictions=[]
        for x in X:
            predictions.append(self._predict(x))
        return np.array(predictions)

    def score(self,X,Y):
        predictions=self.predict(X)
        return np.mean(predictions==Y)