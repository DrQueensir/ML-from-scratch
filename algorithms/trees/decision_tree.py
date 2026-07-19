import numpy as np

class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right 
        self.value=value 

class DecisionTree:
    def __init__(self,max_depth=100,min_samples_split=2):
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.root=None

    def fit(self,X,Y):
        self.root=self._grow_tree(X,Y)
        return self

    def predict(self,X):
        return np.array([self._traverse_tree(x,self.root) for x in X])

    def score(self,X,Y):
        predictions=self.predict(X)
        return np.mean(predictions==Y)

    def _grow_tree(self,X,Y,depth=0):
        samples,features=X.shape
        num_classes=len(np.unique(Y))
        if((depth>=self.max_depth) or num_classes==1 or samples<self.min_samples_split):
            leaf=self._most_common_label(Y)
            return Node(value=leaf)
        
        (feature,threshold,left,right)=self._best_split(X,Y)

        if feature is None:
            leaf=self._most_common_label(Y)
            return Node(value=leaf)
        left_X=X[left]
        left_Y=Y[left]
        right_X=X[right]
        right_Y=Y[right]

        left_child=self._grow_tree(left_X,left_Y,depth+1)
        right_child=self._grow_tree(right_X,right_Y,depth+1)

        return Node(feature=feature,threshold=threshold,left=left_child,right=right_child)

    def _best_split(self,X,Y):
        _,features=X.shape
        best_gain=-1
        best_feature=None
        best_threshold=None
        best_left_mask = None
        best_right_mask = None
        for i in range(features):
            column=X[:,i]
            unique_values=np.unique(column)
            for j in range(len(unique_values)-1):
                threshold=(unique_values[j]+unique_values[j+1])/2
                left= column< threshold
                right=column>=threshold
                left_y=Y[left]
                right_y=Y[right]
                if len(left_y)==0 or len(right_y)==0: continue
                ig=self._information_gain(Y,left_y,right_y)
                if ig>best_gain:
                    best_gain=ig
                    best_feature=i
                    best_threshold=threshold
                    best_left_mask=left
                    best_right_mask=right
        return(best_feature,best_threshold,best_left_mask,best_right_mask)



    def _information_gain(self,parent,left,right):
        parent_entropy=self._entropy(parent)

        left_entropy=self._entropy(left)
        right_entropy=self._entropy(right)
        lweight=len(left)/len(parent)
        rweight=len(right)/len(parent)

        child_entropy=(left_entropy*lweight+right_entropy*rweight)

        return (parent_entropy-child_entropy)
    
    def _entropy(self,Y):
        _,counts=np.unique(Y,return_counts=True)
        probabilities=counts/len(Y)
        entropy=-np.sum(probabilities*np.log2(probabilities))
        return entropy
    
    def _most_common_label(self,Y):
        classes,counts=np.unique(Y,return_counts=True)
        index=np.argmax(counts)
        return classes[index]
    
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature]< node.threshold:
            return self._traverse_tree(x,node.left)
        else:
            return self._traverse_tree(x,node.right)
