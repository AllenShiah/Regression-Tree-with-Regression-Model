

import numpy as  np
from sklearn.linear_model import LinearRegression # regression model used to predict segmented groups

class Node:
    '''
    nodel class is used as linked list to store information of each node
    '''
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):

        self.left = left # store next class in the left
        self.right = right # store next class in the right
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.var_red = var_red
        # for leaf node
        self.value = value


class DecisionTreeRegression():
    '''
    This is the class used to train model and predict data
    '''
    def __init__(self, max_depth=2, min_samples_split=3, kernel = 'regression'):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.kernel = kernel

    def fit(self, X, Y):
        ''' 
        function to train the tree (designs refered to sklearn)
        '''
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def build_tree(self, dataset, curr_depth=0):
        ''' 
        use recursion to build the tree 
        '''
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        best_split = {}
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["var_red"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["var_red"])
        
        # compute leaf node
        leaf_model = self.calculate_leaf_value(X, Y)
        # return leaf node
        return Node(value=leaf_model)

    def get_best_split(self, dataset, num_samples, num_features):
        ''' 
        function to find the best split 
        '''
        # dictionary to store the best split
        best_split = {}
        max_var_red = -float("inf") 

        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    
                    # compute information gain
                    curr_var_red = self.variance_reduction(dataset, dataset_left, dataset_right, self.kernel)
                    # update the best split if needed
                    if curr_var_red > max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red
                        
        # return best split
        return best_split

    def split(self, dataset, feature_index, threshold):
        ''' 
        function to split the data 
        '''
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def variance_reduction(self, dataset, dataset_left, dataset_right, kernel):
        ''' 
        function to compute variance reduction 
        '''
        if kernel == 'regression':
            X, Y = dataset[:,:-1], dataset[:,-1]
            X_left, Y_left, X_right, Y_right = dataset_left[:,:-1], dataset_left[:,-1], dataset_right[:,:-1], dataset_right[:,-1]
            model = LinearRegression().fit(X= X, y=Y)
            model_left = LinearRegression().fit(X=X_left, y=Y_left)
            model_right = LinearRegression().fit(X=X_right, y=Y_right)
            SSE = np.sum(np.power((Y-model.predict(X)),2))
            SSE_left = np.sum(np.power((Y_left - model_left.predict(X_left)), 2))
            SSE_right = np.sum(np.power((Y_right - model_right.predict(X_right)), 2))
            weight_l = len(l_child) / len(parent)
            weight_r = len(r_child) / len(parent)
            reduction = SSE - (weight_l*SSE_left + weight_r*SSE_right)
            return reduction

        elif kernel == 'average':
            parent, l_child, r_child = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
            weight_l = len(l_child) / len(parent)
            weight_r = len(r_child) / len(parent)
            reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
            return reduction
        
    def calculate_leaf_value(self, X, Y):
        ''' 
        function to compute leaf node 
        '''
        if self.kernel == 'regression':
            model = LinearRegression().fit(X, Y)
            return model
        elif self.kernel == 'average':
            val = np.mean(Y)
            return val

    def predict(self, X):
        ''' function to predict a single data point '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        ''' 
        function to predict new dataset (recursive)
        '''
        
        if tree.value!=None:
            if self.kernel == 'regression': 
                return tree.value.predict(np.array([x]))
            elif self.kernel == 'average':
                return tree.value

        feature_val = x[tree.feature_index]

        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    




