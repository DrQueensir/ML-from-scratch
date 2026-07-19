from decision_tree import DecisionTree
import numpy as np

class RandomForest:

    def __init__(self,n_estimators=100,max_depth=None,min_samples_split=2,max_features=None):
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.max_features=max_features
        self.trees=[]

    def fit(self,X,Y):
        self.trees =[]

        for _ in range(self.n_estimators):
            X_bootstrap,Y_bootstrap=self._bootstrap_sample(X,Y)
            tree=DecisionTree(self.max_depth,self.min_samples_split)
            tree.fit(X_bootstrap,Y_bootstrap)
            self.trees.append(tree)
        
        return self

    def predict(self,X):
        predictions=np.array([tree.predict(X) for tree in self.trees])
        predictions=predictions.T
        return np.array([self._most_common_label(row) for row in predictions])
    
    def score(self,X,Y):
        predictions=self.predict(X)
        return np.mean(predictions==Y)

    def _bootstrap_sample(self, X, Y):
        samples=X.shape[0]

        indices=np.random.choice(samples,size=samples,replace=True)

        X_bootstrap=X[indices]
        Y_bootstrap=Y[indices]

        return X_bootstrap,Y_bootstrap
    
    def _most_common_label(self,Y):
        classes,counts=np.unique(Y,return_counts=True)
        index=np.argmax(counts)
        return classes[index]
