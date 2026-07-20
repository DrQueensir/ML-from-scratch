import numpy as np
from regression_tree import RegressionTree

class GradientBoosting:

    def __init__(self,n_estimators=100,learningrate=0.1,max_depth=3,min_samples_split=2):
        self.n_estimators=n_estimators
        self.learningrate=learningrate
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.initial_prediction=None
        self.trees=[]

    def fit(self,X,Y):
        self.trees=[]
        self.initial_prediction = np.mean(Y)
        predictions=np.full(Y.shape,self.initial_prediction)

        for _ in range(self.n_estimators):
            y_error=Y-predictions
            tree=RegressionTree(self.max_depth,self.min_samples_split)
            tree.fit(X,y_error)
            self.trees.append(tree)
            tree_predictions=tree.predict(X)
            predictions= predictions +(self.learningrate*tree_predictions)

        return self


    def predict(self,X):
        predictions=np.full(X.shape[0],self.initial_prediction)
        for tree in self.trees:
            tree_prediction=tree.predict(X)
            predictions=predictions+(self.learningrate*tree_prediction)
        return predictions

    def score(self,X,Y):
        predictions = self.predict(X)
        rss=np.sum((Y-predictions)**2)
        tss=np.sum((Y-np.mean(Y))**2)
        return float((1-(rss/tss)))
