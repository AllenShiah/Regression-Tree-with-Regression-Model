
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import RegressionTree as RT

#%%



def main(X_train, X_test, Y_train, Y_test):
    mse_list = list()
    mse_list2 = list()
    depth_start = 2
    depth_end = 15
    for depth in range(depth_start, depth_end):
        regressor = RT.DecisionTreeRegression(min_samples_split=3, max_depth=depth, kernel='regression')
        regressor.fit(X_train,Y_train)
        Y_pred = regressor.predict(X_test) 
        mse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        mse_list.append(mse)

        regressor2 = RT.DecisionTreeRegression(min_samples_split=3, max_depth=depth, kernel='average')
        regressor2.fit(X_train,Y_train)
        Y_pred2 = regressor2.predict(X_test) 
        mse2 = np.sqrt(mean_squared_error(Y_test, Y_pred2))
        mse_list2.append(mse2)

    plt.plot(np.arange(depth_start, depth_end), mse_list, label="regression")
    plt.plot(np.arange(depth_start, depth_end), mse_list2, label = "average")
    plt.xlabel("depth")
    plt.ylabel("RMSE")
    plt.legend()



if __name__=='__main__':
    data = pd.read_csv("airfoil_noise_data.csv")
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1,1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
    main(X_train, X_test, Y_train, Y_test)

# %%
