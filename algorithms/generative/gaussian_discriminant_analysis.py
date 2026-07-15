import numpy as np

class GaussianDiscriminantAnalysis:

    def __init__(self):
        self.classes=None
        self.means=None
        self.covariance=None
        self.priors=None
    
    def _multivariate_gaussian(self,x,mean,covariance):
        exponent=np.exp((-1/2)*((x-mean).T)@(np.linalg.inv(covariance))@(x-mean))
        coefficient=1/(((2*np.pi)**(len(x)/2))*(np.sqrt(np.linalg.det(covariance))))
        return exponent*coefficient

    def fit(self,X,Y):
        samples,features=X.shape
        self.classes=np.unique(Y)
        num_classes=len(self.classes)
        self.means=np.zeros((num_classes,features))
        self.priors=np.zeros(num_classes)
        self.covariance=np.zeros((features,features))

        for index,c in enumerate(self.classes):
            X_class=X[Y==c]
            self.means[index]=np.mean(X_class,axis=0)
            self.priors[index]=len(X_class)/samples
            difference=X_class-self.means[index]
            self.covariance= self.covariance + difference.T @ difference

        self.covariance=self.covariance/samples

    def _predict(self,x):
        posteriors=[]
        for index,c in enumerate(self.classes):
            prior=np.log(self.priors[index])
            likelihood=self._multivariate_gaussian(x,self.means[index],self.covariance)
            log_likelihood=np.log(likelihood)
            posterior=prior+log_likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self,X):
        predictions=[]
        for x in X:
            predictions.append(self._predict(x))
        return np.array(predictions)

    def score(self,X,Y):
        predictions=self.predict(X)
        return np.mean(predictions==Y)