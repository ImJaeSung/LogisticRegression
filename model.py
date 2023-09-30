#%%
import numpy as np
#%%
class LogisticRegression():
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
    
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def cross_entropy_loss(self, y, y_hat):
        delta = 1e-7
        n = len(y)
        return np.sum(-(y*np.log(y_hat + delta) + (1 - y)*np.log(1 - y_hat + delta)))/n 
       
    def gradient(self, x, y, y_hat):
        n = len(y)
        grad_w = - np.dot(x.transpose(), y-y_hat)/n
        grad_b = - (y-y_hat)/n
        return grad_w, grad_b
    
    def fit(self, x, y):

        self.w = np.zeros(x.shape[1])
        self.b = np.zeros(x.shape[0])
        self.losses = []

        for epoch in range(self.epochs):
            z = np.dot(x, self.w) + self.b
            y_hat = self.sigmoid(z)
            
            grad_w, grad_b = self.gradient(x, y, y_hat)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            z = np.dot(x, self.w) + self.b
            y_hat = self.sigmoid(z)

            loss = self.cross_entropy_loss(y, y_hat)
            self.losses.append(loss)
            if epoch % 1000 == 0:
                print(f'loss: {loss} \t')
    
    def predict_prob(self, x):
        return self.sigmoid(np.dot(x, self.w)) + self.b

    def predict(self, x):
        return self.predict_prob(x).round()
