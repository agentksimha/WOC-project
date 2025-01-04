import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def split_dataset(x,y,n):
    x_train = np.array(x[0:int(np.ceil((1-n)*len(x)))])
    x_cv = np.array(x[int(np.ceil((1-n)*len(x))):int((0.5*(len(x)+np.ceil((1-n)*len(x)))))])
    x_test = np.array(x[int((0.5*(len(x)+np.ceil((1-n)*len(x))))):len(x)])
    y_train = np.array(y[0:int(np.ceil((1-n)*len(x)))])
    y_cv = np.array(y[int(np.ceil((1-n)*len(x))):int((0.5*(len(x)+np.ceil((1-n)*len(x)))))])
    y_test = np.array(y[int((0.5*(len(x)+np.ceil((1-n)*len(x))))):len(x)])
    print("x_train,x_cv,x_test,y_train,y_cv,y_test :",x_train,x_cv,x_test,y_train,y_cv,y_test)
    return x_train,x_cv,x_test,y_train,y_cv,y_test
    

def svlr(x,y,epochs,L):
    global n
    n = len(x)
    def total_error(x,y,m,b):
        total_error += 1/2*n*np.dot((y-(m*x+b)),(y-(m*x+b)))
        return total_error
    def gradient_decesent(m_now,b_now,x,y,L):
        m_gradient = 0
        b_gradient = 0
        m_gradient = (-1/n)*np.dot(x,(y-(m_now*x+b_now)))
        b_gradient = (-1/n)*np.sum((y-(m_now*x+b_now)))
        global m,b
        m = m_now - L*m_gradient
        b = b_now - L*b_gradient
        return m,b
    m=0
    b=0
    
    
    for i in range(epochs):
        m,b = gradient_decesent(m,b,x,y,L)

    model = m*x+b
    print(f"slope of graph:{m}")
    print(f"intercept of graph:{b}")
    global e
    e = np.sum(((y-(model)))**2)/n
    plt.scatter(x,y)
    plt.plot(x,model)
    plt.show()
    print(f" training error :{e}")
def single_predict(X,x_cv,y_test,y_cv):
    yhat = m*X+b
    yhat1 = m*x_cv+b
    g = np.sum(abs((y_test-yhat)))/n
    print(f"test error :{g}")
    h = np.sum(abs((y_cv-yhat1)))/n
    print(f"cross validation error :{h}")

class LogisticRegression():
    def __init__(self,learning_rate,iterations):
        self.learning_rate  = learning_rate
        self.iterations = iterations
    def fit(self,x,y):
            n = len(x)
            self.x = x
            self.y = y
            self.b = 0
            self.m = 0
            for i in range(self.iterations):
                self.update_weights()
                return self
    def update_weights(self):
        global p 
            
            
     
        p = 1/(1+np.exp(-(np.dot(self.m,self.x)+self.b)))
        dm = (1/n)*np.dot(p-self.y,self.x)
        db = (1/n)*np.sum(p-self.y)
        self.m = self.m - self.learning_rate*dm - self.learning_rate
        self.b = self.b - self.learning_rate*db
        global r
        r = 1/(1+np.exp(-(np.dot(self.m,self.x)+self.b)))
        global window_size
        window_size =5
        global r_smooth
        r_smooth = np.convolve(r,np.ones(window_size)/window_size,mode='valid')
        return self.m ,self.b,p
def predict(self,X):
        Z = 1/(1+np.exp(-(np.dot(self.m,X)+self.b)))
        Y = np.where(Z>0.5,1,0)
        print(self.m,self.b)
        X = np.linspace(np.amin(self.x),np.amax(self.x),len(self.y))
        plt.scatter(X,self.y,alpha=0.5)
        plt.plot(X[2:-2],r_smooth)
        plt.show()


def pr(x,y,n):
     model = np.poly1d(np.polyfit(x,y,n))
     global myline
     myline = np.linspace(np.amin(x),np.amax(x),len(y))
     global coefficients
     coefficients = model.coeffs
     global p
     p =len(coefficients)
     r_2 = 1-((np.sum(y-model(myline))**2)/(np.sum((y-np.mean(y))**2)))
     print("r squared error:" , r_2)
    
     plt.scatter(x,y)
     plt.plot(myline,model(myline),linewidth = 10,color = 'red')
     plt.show()

def predict_pr(X):
   
     p = len(coefficients)
     z = np.zeros(p)
     for i in range(len(z)):
      z[i] = X**i

     print(z)

    

     for  j in range(len(z)):
      for k in range(len(z)):
        if j<k:
            z[j],z[k] = z[k],z[j]


            print(z)
            predict = np.dot(coefficients,z)
            print(predict)
def pass_testset(X):
    predict = np.zeros(len(X))
    for i in range(len(X)):
        predict[i] = predict_pr(X[i])


    print("targets for test set ",predict)
 








def euclidean_distance(x1,x2):
    return np.sqrt(np.sum(x1-x2)**2)

class knn():
    def __init__(self,k=3):
        self.k  = k
    def fit(self,X,y):
        self.X_train =X
        self.y_train = y
    def predict(self,X):
        predictions  = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self,x):
        distance = [euclidean_distance(x,x_train) for x_train in self.X_train]
        k_indices=np.argsort(distance)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = pd.Series(k_nearest_labels).value_counts().idxmax
        print(most_common)
    def calculate_f1_score(self, X_test, y_test):
       
        y_pred = self.predict(X_test)

        classes = np.unique(y_test)
        f1_scores = []

        for cls in classes:
       
            tp = np.sum((y_pred == cls) & (y_test == cls))
            fp = np.sum((y_pred == cls) & (y_test != cls))
            fn = np.sum((y_pred != cls) & (y_test == cls))

  
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

   
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            f1_scores.append(f1)

   
        weights = [np.sum(y_test == cls) / len(y_test) for cls in classes]
        weighted_f1 = np.sum(np.array(f1_scores) * np.array(weights))
        return weighted_f1
 





class kmc():

    def __init__(self, k=3, iterations=100, plot_step=False):
        self.k = k
        self.iterations = iterations
        self.plot_step = plot_step
        self.clusters = [[] for _ in range(self.k)]  
        self.centroids = None

    def predict(self, x):
        self.x = x if isinstance(x, np.ndarray) else x.to_numpy() 
        self.n_samples, self.n_features = self.x.shape
        
        random_sample_indices = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = self.x[random_sample_indices]

        for _ in range(self.iterations):
            self.clusters = self._create_clusters(self.centroids)
            
            if self.plot_step:
                self.plot()
            
            centroids_old = self.centroids
            
            self.centroids = self._get_centroids(self.clusters)
            
            if self.converged(centroids_old, self.centroids):
                break

            if self.plot_step:
                self.plot()

        return self._get_cluster_labels(self.clusters)

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.x):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            if cluster:  
                cluster_mean = np.mean(self.x[cluster], axis=0)
                centroids[cluster_idx] = cluster_mean
        return centroids

    def converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.k)]
        return sum(distances) == 0

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples, dtype=int)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels
    def plot(self):
        plt.figure(figsize=(8, 6))
        for cluster_idx, cluster in enumerate(self.clusters):
            cluster_points = self.x[cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_idx + 1}")
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=200, c='black', marker='X', label='Centroids')
        plt.title("K-Means Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()
    
    def davies_bouldin_index(self):
        if self.centroids is None or not self.clusters:
            raise ValueError("Model must be fitted before calculating Davies-Bouldin Index.")

        cluster_distances = []
        for i, cluster in enumerate(self.clusters):
            if len(cluster) > 0:
                intra_distance = np.mean([euclidean_distance(self.x[idx], self.centroids[i]) for idx in cluster])
                cluster_distances.append(intra_distance)
            else:
                cluster_distances.append(0)

        db_index = 0
        for i in range(self.k):
            max_ratio = 0
            for j in range(self.k):
                if i != j:
                    inter_distance = euclidean_distance(self.centroids[i], self.centroids[j])
                    if inter_distance > 0:
                        ratio = (cluster_distances[i] + cluster_distances[j]) / inter_distance
                        max_ratio = max(max_ratio, ratio)
            db_index += max_ratio

        return db_index / self.k
    





            
def dense(a_in,units,linear = None,relu  =None,sigmoid = None):
        global W,b
        W  = np.random.rand(a_in.shape[0],units)
        b = np.random.rand(units,1)
        a_out = np.zeros(units)
        z = np.zeros(units)
        sof = np.zeros(units)
        for i in range(units):
            w = W[:,i]
            z[i] = np.dot(w,a_in)+b[i]
            a = 0
            for j in range(units):
                a += np.exp(z[j])
                sof[i] = np.exp(z[i])/a
            print(f"linear:{linear}")
            print(f"relu:{relu}")
            print(f"sigmoid:{sigmoid}")
      
           
            
            
            print(z)
            
        my_dict = {'linear': z ,'relu':np.maximum(0,z), 'sigmoid' : ((1/(1+np.exp(-z)))) , 'softmax' : sof}
        a_out = my_dict
        print(a_out)

        
def dense2(a_in,units,linear = None,relu  =None,sigmoid = None):
        W  = np.random.rand(a_in.shape[0],units)
        b = np.random.rand(units,1)
        a_out = np.zeros(units)
        z = np.zeros(units)
        sof = np.zeros(units)
        for i in range(units):
            w = W[:,i]
            z = np.matmul(w,a_in)+b[i]
            sof = np.exp(z) / (np.sum(np.exp(z)))

            print(f"linear:{linear}")
            print(f"relu:{relu}")
            print(f"sigmoid:{sigmoid}")
            
            print(z)
        global my_dict    
        my_dict = {'linear': z ,'relu':np.maximum(0,z), 'sigmoid' : ((1/(1+np.exp(-z)))) ,"softmax" : sof}
        a_out = my_dict
        print(a_out)
        


   

def dense3(data,key):
    print(f"full dict:{data}")
    print(f"value of {key}:",data.get(key,'key not found'))
    return data[key]






class Sequential():
    def __init__(self,n,units):
        self.n = n

        self.units = units
       
    def fit(self,x,y , epochs):
        self.x  =x
        self.y  = y
        self.epochs = epochs
    
    def connect_layers(self):
        self.units = list(self.units)
 
        z = [0]*self.n
        for i in range(self.n):
            z[i] = [0]*self.units[i]
            print(z)
            for j in range(1,self.n):
                z[0] = self.x
                z[j] = dense(z[j-1],self.units[j],linear = None,relu  =None,sigmoid = None)

                z[j] = dense3(my_dict,'relu')

                
               
                print("layer :",self.n ,'output of respective layer:',z[self.n])
            





            print("ouputs from  all layers : ",z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def sigmoid_derivative(a):
    return a * (1 - a)


def backpropagate(z, my_dict, y_true, learning_rate,W,b):
    

    a_out = my_dict['softmax']  
    dz = a_out - y_true  

    for i in reversed(range(len(z))):  
        da = dz
        if i > 0: 
            activation = my_dict['relu'] if i < len(z) - 1 else my_dict['softmax']
            dz = da * relu_derivative(activation)  
            dz = da 
        
        w_update = np.dot(z[i - 1].T, dz)
        b_update = np.sum(dz, axis=0, keepdims=True)
        
        W -= learning_rate * w_update
        b -= learning_rate * b_update
        
        # Calculate error for the next layer
        dz = np.dot(dz, W.T)




















































































































































































































def mvlr(x, y, epochs, L):
    global n, m
    n = len(x)

    def total_error(x, y, m):
        x = np.hstack((np.ones((x.shape[0], 1)), x))  
        predictions = np.matmul(x, m)
        errors = y - predictions
        mse = (1 / (2 * n)) * np.sum(errors**2)
        return mse

    def gradient_descent(m_now, x, y, L):
        x = np.hstack((np.ones((x.shape[0], 1)), x)) 
        y = y.reshape(-1, 1)
        m_now = m_now.reshape(-1, 1)
        gradients = (-1 / n) * np.dot(x.T, (y - np.dot(x, m_now)))
        m_next = m_now - L * gradients
        return m_next

 
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

 
    m = np.zeros((x.shape[1] + 1, 1))

    for i in range(epochs):
        m = gradient_descent(m, x, y, L)
       

def predict_linr(X, x_cv, x_train, y_test, y_cv, y_train):
    X = np.hstack((np.ones((X.shape[0], 1)), X))  
    x_cv = np.hstack((np.ones((x_cv.shape[0], 1)), x_cv))
    x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))

    yhat = np.matmul(X, m)
    yhat1 = np.matmul(x_cv, m)
    yhat2 = np.matmul(x_train, m)

    mse = np.mean((y_test - yhat)**2)
    mse1 = np.mean((y_cv - yhat1)**2)
    mse2 = np.mean((yhat2 - y_train)**2)

    print("Predictions:", yhat.flatten())
    print("MSE on Test:", mse)
    print("MSE on Validation:", mse1)
    print("MSE on Training:", mse2)

























































































class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        X = self._one_hot_encode(X)
        self.tree = self._build_tree(X, y)

    def predict(self, X):
   
        X = self._one_hot_encode(X)
        return np.array([self._predict(sample, self.tree) for sample in X])

    def _entropy(self, y):
 
        unique, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs))

    def _split(self, X, y, feature_index, threshold):
     
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _best_split(self, X, y):
        
        best_feature = None
        best_threshold = None
        best_entropy = float('inf')
        best_splits = None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                entropy = (len(y_left) * self._entropy(y_left) + len(y_right) * self._entropy(y_right)) / len(y)

                if entropy < best_entropy:
                    best_feature = feature_index
                    best_threshold = threshold
                    best_entropy = entropy
                    best_splits = (X_left, X_right, y_left, y_right)

        return best_feature, best_threshold, best_splits

    def _build_tree(self, X, y, depth=0):
      
        if len(np.unique(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return {'leaf': True, 'label': np.bincount(y).argmax()}

        best_feature, best_threshold, best_splits = self._best_split(X, y)
        if not best_splits:
            return {'leaf': True, 'label': np.bincount(y).argmax()}

        X_left, X_right, y_left, y_right = best_splits
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(X_left, y_left, depth + 1),
            'right': self._build_tree(X_right, y_right, depth + 1),
        }

    def _predict(self, sample, node):
       
        if node['leaf']:
            return node['label']
        if sample[node['feature']] <= node['threshold']:
            return self._predict(sample, node['left'])
        else:
            return self._predict(sample, node['right'])

    def _one_hot_encode(self, X):
       
        X_encoded = []
        for col in range(X.shape[1]):
            unique_vals = np.unique(X[:, col])
            if len(unique_vals) > 10:  
                X_encoded.append(X[:, col].reshape(-1, 1))
            else:  
                encoded = (X[:, col].reshape(-1, 1) == unique_vals).astype(float)
                X_encoded.append(encoded)
        return np.hstack(X_encoded)
    def calculate_f1_score(self, X_test, y_test):
        
        y_pred = self.predict(X_test)

        classes = np.unique(y_test)
        f1_scores = []

        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_test == cls))
            fp = np.sum((y_pred == cls) & (y_test != cls))
            fn = np.sum((y_pred != cls) & (y_test == cls))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            f1_scores.append(f1)
        weights = [np.sum(y_test == cls) / len(y_test) for cls in classes]
        weighted_f1 = np.sum(np.array(f1_scores) * np.array(weights))
        return weighted_f1





































































































































































    


































class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
    
        X = self._one_hot_encode(X)
        self.tree = self._build_tree(X, y)

    def predict(self, X):

        X = self._one_hot_encode(X)
        return np.array([self._predict(sample, self.tree) for sample in X])

    def _entropy(self, y):
        unique, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs))

    def _split(self, X, y, feature_index, threshold):
 
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_entropy = float('inf')
        best_splits = None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

            
                entropy = (len(y_left) * self._entropy(y_left) + len(y_right) * self._entropy(y_right)) / len(y)

                if entropy < best_entropy:
                    best_feature = feature_index
                    best_threshold = threshold
                    best_entropy = entropy
                    best_splits = (X_left, X_right, y_left, y_right)

        return best_feature, best_threshold, best_splits

    def _build_tree(self, X, y, depth=0):

        if len(np.unique(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return {'leaf': True, 'label': np.bincount(y).argmax()}

        best_feature, best_threshold, best_splits = self._best_split(X, y)
        if not best_splits:
            return {'leaf': True, 'label': np.bincount(y).argmax()}

        X_left, X_right, y_left, y_right = best_splits
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(X_left, y_left, depth + 1),
            'right': self._build_tree(X_right, y_right, depth + 1),
        }

    def _predict(self, sample, node):
   
        if node['leaf']:
            return node['label']
        if sample[node['feature']] <= node['threshold']:
            return self._predict(sample, node['left'])
        else:
            return self._predict(sample, node['right'])

    def _one_hot_encode(self, X):
    
        X_encoded = []
        for col in range(X.shape[1]):
            unique_vals = np.unique(X[:, col])
            if len(unique_vals) > 10: 
                X_encoded.append(X[:, col].reshape(-1, 1))
            else: 
                encoded = (X[:, col].reshape(-1, 1) == unique_vals).astype(float)
                X_encoded.append(encoded)
        return np.hstack(X_encoded)
    




def mvlr(x, y, epochs, L):
    global n, m
    n = len(x)

    def total_error(x, y, m):
        x = np.hstack((np.ones((x.shape[0], 1)), x))  
        predictions = np.matmul(x, m)
        errors = y - predictions
        mse = (1 / (2 * n)) * np.sum(errors**2)
        return mse

    def gradient_descent(m_now, x, y, L):
        x = np.hstack((np.ones((x.shape[0], 1)), x)) 
        y = y.reshape(-1, 1)
        m_now = m_now.reshape(-1, 1)
        gradients = (-1 / n) * np.dot(x.T, (y - np.dot(x, m_now)))
        m_next = m_now - L * gradients
        return m_next

 
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

 
    m = np.zeros((x.shape[1] + 1, 1))

    for i in range(epochs):
        m = gradient_descent(m, x, y, L)
       

def predict_linr(X, x_cv, x_train, y_test, y_cv, y_train):
    X = np.hstack((np.ones((X.shape[0], 1)), X))  
    x_cv = np.hstack((np.ones((x_cv.shape[0], 1)), x_cv))
    x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))

    yhat = np.matmul(X, m)
    yhat1 = np.matmul(x_cv, m)
    yhat2 = np.matmul(x_train, m)

    mse = np.mean((y_test - yhat)**2)
    mse1 = np.mean((y_cv - yhat1)**2)
    mse2 = np.mean((yhat2 - y_train)**2)

    print("Predictions:", yhat.flatten())
    print("MSE on Test:", mse)
    print("MSE on Validation:", mse1)
    print("MSE on Training:", mse2)







        




        

    










     

