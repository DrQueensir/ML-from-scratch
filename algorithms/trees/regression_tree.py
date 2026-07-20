import numpy as np

class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right 
        self.value=value 

class RegressionTree:

    def __init__(self,max_depth=100,min_samples_split=2):
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.root=None

    def fit(self,X,Y):
        self.root=self._grow_tree(X,Y,depth=0)
        return self
    
    def predict(self,X):
        return np.array([self._traverse_tree(x,self.root) for x in X])
    
    def score(self,X,Y):
        predictions=self.predict(X)
        rss=np.sum((Y-predictions)**2)
        tss=np.sum((Y-np.mean(Y))**2)
        return float((1-(rss/tss)))

    def _grow_tree(self,X,Y,depth):
        samples,features=X.shape
        if((depth>=self.max_depth) or (np.var(Y)< 1e-12) or (samples<self.min_samples_split)):
            leaf=np.mean(Y)
            return Node(value=leaf)
        
        (feature,threshold,left,right)=self._best_split(X,Y)

        if(feature is None):
            leaf=np.mean(Y)
            return Node(value=leaf)
        
        X_left=X[left]
        Y_left=Y[left]
        X_right=X[right]
        Y_right=Y[right]

        left_child=self._grow_tree(X_left,Y_left,depth+1)
        right_child=self._grow_tree(X_right,Y_right,depth+1)

        return Node(feature=feature,threshold=threshold,left=left_child,right=right_child)

    def _best_split(self,X,Y):
        _,features=X.shape
        best_variance=-1
        best_feature=None
        best_threshold=None
        best_left_mask= None
        best_right_mask=None

        for i in range(features):
            column=X[:,i]
            unique_values=np.unique(column)
            for j in range(len(unique_values)-1):
                threshold=(unique_values[j]+unique_values[j+1])/2
                left=column<threshold
                right=column>=threshold

                left_y=Y[left]
                right_y=Y[right]

                if(len(left_y)==0 or len(right_y)==0):continue

                var_reduce=self._variance_reduction(Y,left_y,right_y)

                if(var_reduce>best_variance):
                    best_variance=var_reduce
                    best_feature=i
                    best_threshold=threshold
                    best_left_mask=left
                    best_right_mask=right

        return (best_feature,best_threshold,best_left_mask,best_right_mask)

    def _variance_reduction(self,parent,left,right):
        parent_var=self._variance(parent)
        left_var=self._variance(left)
        right_var=self._variance(right)

        left_weight=len(left)/len(parent)
        right_weight=len(right)/len(parent)

        child_var=((left_weight*left_var)+(right_weight*right_var))

        return (parent_var-child_var)

    def _variance(self,Y):
        return np.var(Y)

    def _traverse_tree(self,x,node):

        if node.value is not None:
            return node.value
        
        if x[node.feature]<node.threshold:
            return self._traverse_tree(x,node.left)
        else:
            return self._traverse_tree(x,node.right)