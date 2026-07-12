import numpy as np 

class LinearRegression:
    def __init__(self, learningrate=0.01,iterations=1000,optimizer="batch",batch_size=32):
        self.weights=None
        self.bias=None

        self.learningrate=learningrate
        self.iterations=iterations
        self.optimizer=optimizer
        self.batch_size=batch_size

        self.cost_history=[]

    def fit(self,X,Y):
        features=X.shape[1]
        self.weights=np.zeros(features)
        self.bias=0.0
        self.cost_history=[]

        if self.optimizer=="batch":
            self._batch_gradient_descent(X,Y)
        elif self.optimizer=="sgd":
            self._stochastic_gradient_descent(X,Y)
        elif self.optimizer=="mini-batch":
            self._mini_batch_gradient_descent(X,Y)
        else:
            raise ValueError("Unknown optimizer")
    
    def _batch_gradient_descent(self,X,Y):
        samples=X.shape[0]

        for _ in range(self.iterations):
            y_pred=np.dot(X,self.weights)+self.bias
            y_error=y_pred-Y
            cost = (1 / (2 * samples)) * np.sum(y_error ** 2)
            self.cost_history.append(cost)

            dw=(1/samples)*np.dot(X.T,y_error)
            db=(1/samples)*np.sum(y_error)

            self.weights=self.weights - (self.learningrate*dw)
            self.bias=self.bias - (self.learningrate*db)
    
    def _stochastic_gradient_descent(self,X,Y):
        samples=X.shape[0]
        indices = np.random.permutation(samples)
        for i in indices:
            for i in range(samples):
                x_i=X[i]
                y_i=Y[i]
                y_pred=np.dot(x_i,self.weights)+self.bias
                y_error=y_pred-y_i

                dw=x_i*y_error
                db=y_error

                self.weights=self.weights - (self.learningrate*dw)
                self.bias=self.bias - (self.learningrate*db)
            
            prediction=np.dot(X,self.weights)+self.bias
            error=prediction-Y
            cost = (1 / (2 * samples)) * np.sum(error ** 2)
            self.cost_history.append(cost)
    
    def _mini_batch_gradient_descent(self,X,Y):
        samples=X.shape[0]

        for _ in range(self.iterations):
            indices= np.random.permutation(samples)
            X_shuffled=X[indices]
            Y_shuffled=Y[indices]

            for start in range(0,samples,self.batch_size):
                end=start+self.batch_size
                X_batch=X_shuffled[start:end]
                Y_batch=Y_shuffled[start:end]

                y_pred=np.dot(X_batch,self.weights)+self.bias
                y_error=y_pred-Y_batch

                batch_size_actual = X_batch.shape[0]

                dw=(1/batch_size_actual)*np.dot(X_batch.T,y_error)
                db=(1/batch_size_actual)*np.sum(y_error)

                self.weights=self.weights - (self.learningrate*dw)
                self.bias=self.bias - (self.learningrate*db)

            prediction=np.dot(X,self.weights)+self.bias
            error=prediction-Y
            cost=(1/(2*samples))*np.sum(error**2)
            self.cost_history.append(cost)


    def predict(self,X):
        return np.dot(X,self.weights)+self.bias
    
    def score(self, X, Y):
        predictions = self.predict(X)

        ss_res = np.sum((Y - predictions)**2)
        ss_tot = np.sum((Y - np.mean(Y))**2)

        return 1 - (ss_res / ss_tot)