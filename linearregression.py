import matplotlib.pyplot as plt
import numpy as np
import math
def linear_regression(X, y, numIterations, eta):
    m = X.shape[0];
    s = (1, 2)
    theta = np.ones(s);
    X_transpose = X.transpose();
    for iter in range(numIterations):
        sumtheta1 = 0;
        sumtheta2 = 0;
        sum2 = 0;

        for i in range(m):
            dif = theta[0][0] + theta[0][1]*X[i] - y[i];
            sumtheta1 = sumtheta1 + dif;
            sumtheta2 = sumtheta2 + dif*X[i];
            sum2 = sum2 + pow(dif,2);
        theta[0][0] = theta[0][0] - eta*2/m*sumtheta1;
        theta[0][1] = theta[0][1] - eta*2/m*sumtheta2;
        J = 1/m * sum2;
        print("theta0 : {:.2f} theta1: {:.2f}".format(theta[0][0],theta[0][1]))
        print("MSE = ",J);


    return theta;

m = 100;
X = np.random.rand(m,1) ;
y = 100+3*X+np.random.randn(m,1);
eta = 0.1;
iterNo = 1000;

theta = linear_regression(X,y,iterNo,eta);
print(theta);

plt.scatter(X,y);
y_predicted = theta[0][0] + theta[0][1]*X;
plt.plot(X,y_predicted,c="red");

plt.title("hypothesis ={:.2f}+{:.2f}*X".format(theta[0][0],theta[0][1]));
plt.show();


#


