#### Logistic function(sigmoid function)
$$g:\mathbb{R} \mapsto \mathbb{R}
g(x) = \frac{1}{1+e^{-x}}$$ for $$x\in\mathbb{R}$$
![sigmoid.png](https://www.dropbox.com/scl/fi/t1glpp8jo31eazk013gqe/sigmoid.png?rlkey=vbw7lg30gohq2arjyokqpp9d5&dl=0&raw=1)
```
def sigmoid(self, z):
    return 1/(1 + np.exp(-z))
```

#### Log loss
- $$L(y, \hat{y})=-ylog(\hat{y})-(1-y)log(1-\hat{y})$$
![log_loss.png](https://www.dropbox.com/scl/fi/wt2kegsxe5ysjk22qy4i6/log_loss.png?rlkey=o9nunk1fckhbdxb12nlt80mg1&dl=0&raw=1)

#### Loss function
- $$\mathbf{y} = (y_{1}, y_{2}, \cdots,y_{m})$$ : true value (0 or 1)
- $$\mathbf{\hat{y}} = (\hat{y_{1}}, \hat{y_{2}}, \cdots, \hat{y_{m}})$$ : predictions(probabilities)
- $$C(\mathbf{y}, \mathbf{\hat{y}}) = \frac{1}{m}\sum_{i=1}^{m}L(y_{i}, \hat{y}_{i})$$
- $$\mathbf{x}_{i} = (x_{i, 1},x_{i, 2}, \cdots, x_{i, n})$$
- $$\mathbf{w} = (w_{1},w_{2}, \cdots, w_{n})$$ : weight parameters
- $$b$$ : bias parameter
- $$\hat{y} = g(\mathbf{x}\mathbf{w}+b) = \frac{1}{1+e^-(\mathbf{x}\mathbf{w}+b)}$$
- $$\mathbf{X} = \begin{bmatrix} 
\mathbf{x_{1}} \\ 
\mathbf{x_{2}} \\
\mathbf{x_{3}} \\
\vdots\\
\mathbf{x_{m}} \\
\end{bmatrix} = \begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,n} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,n} \\
x_{3,1} & x_{3,2} & \cdots & x_{3,n} \\
\vdots & \vdots & \ddots & \vdots\\
x_{m,1} & x_{m,2} & \cdots & x_{m,n} \\
\end{bmatrix}$$ 
- $$\mathbf{y} = \begin{bmatrix}
y_{1}\\ 
y_{2} \\
y_{3} \\
\vdots\\
y_{m} \\
\end{bmatrix}$$
- $$\hat{\mathbf{y}} = \begin{bmatrix}
g(\mathbf{x_{1}}\mathbf{w}+b)\\ 
g(\mathbf{x_{2}}\mathbf{w}+b) \\
g(\mathbf{x_{3}}\mathbf{w}+b) \\
\vdots\\
g(\mathbf{x_{m}}\mathbf{w}+b)\\
\end{bmatrix}$$
- $$J(\mathbf{w}, b) \overset{\underset{\mathrm{def}}{}}{=}C(\mathbf{y}, \mathbf{\hat{y}}|\mathbf{X}, \mathbf{w}, b)=\frac{1}{m}\sum_{i=1}^{m}L(y_{i},\frac{1}{1+e^{-(\mathbf{x_{i}}\mathbf{w}+b)}}) = \frac{1}{m}\sum_{i=1}^{m}[-y_{i}log(\frac{1}{1+e^{-(\mathbf{x_{i}}\mathbf{w}+b)}})-(1-y_{i})log(1-\frac{1}{1+e^{-(\mathbf{x_{i}}\mathbf{w}+b)}})]$$
```
def cross_entropy_loss(self, y, y_hat):
    delta = 1e-7
    n = len(y)
    return np.sum(-(y*np.log(y_hat + delta) + (1 - y)*np.log(1 - y_hat + delta)))/n
```
#### Gradient Descent
- $$w_{j}\overset{\underset{\mathrm{def}}{}}{=} w_{j}-\alpha\frac{\partial J(\mathbf{w}, b)}{\partial w_{j}}$$ , for j = 1,2,$$\cdots$$,n
- $$b\overset{\underset{\mathrm{def}}{}}{=} b-\alpha\frac{\partial J(\mathbf{w}, b)}{\partial b}$$
##### weight
- $$\frac{\partial L}{\partial w_{i}} = \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial z} \frac{\partial z}{\partial w_{i}}$$
- $$\frac{\partial L}{\partial \hat{y}} =-(y\frac{1}{\hat{y}}-(1-y)\frac{1}{(1-\hat{y})})$$
- $$\frac{\partial \hat{y}}{\partial z}=\frac{1}{1+e^{-z}}\frac{e^{-z}}{1+e^{-z}} = (\frac{1}{1+e^{-z}})(1-\frac{1}{1+e^{-z}})=\hat{y}(1-\hat{y})$$
- $$\frac{\partial z}{\partial w_{i}} = x_{i}$$
- $$\frac{\partial L}{\partial w_{i}} =-(y\frac{1}{\hat{y}}-(1-y)\frac{1}{(1-\hat{y})})\hat{y}(1-\hat{y})x_{i}=-(y(1-\hat{y})-(1-y)\hat{y})x_{i} = -(y-\hat{y})x_{i}$$
- $$w_{j}\overset{\underset{\mathrm{def}}{}}{=} w_{j}+\alpha(y-\hat{y})x_{i}$$
##### bias
- $$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial z} \frac{\partial z}{\partial b}$$
- $$\frac{\partial z}{\partial b} = 1$$
- $$\frac{\partial L}{\partial b} =-(y\frac{1}{\hat{y}}-(1-y)\frac{1}{(1-\hat{y})})\hat{y}(1-\hat{y})1 =-(\hat{y}-y)1$$
- $$b\overset{\underset{\mathrm{def}}{}}{=} b+\alpha(y-\hat{y})1$$
