import numpy as np

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statistics
import pandas
from sklearn.metrics import r2_score





def normalize_train(X_train):
    mean=np.mean(X_train,axis = 0)
    std=np.std(X_train,axis = 0)
    X=(X_train-mean)/std
    return X, mean, std



def normalize_test(X_test, trn_mean, trn_std):
    X=(X_test-trn_mean)/trn_std
    return X




def get_lambda_range():
    lmbda=np.logspace(-30, 3, num=100)
    return lmbda



def train_model(X, y, l):
    model = Ridge(alpha = l,fit_intercept=True)
    model.fit(X, y)#defalt input
    return model




def train_model_lasso(X,y,l):
    model = Lasso(alpha = l, fit_intercept=True)
    model.fit(X,y)
    return model




def error(X, y, model):
    y_predict = model.predict(X)
    temp = (y-y_predict)**2
    mse = np.mean(temp)
    return mse


def main():
    #Importing dataset
    dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
    dataset_1['Brooklyn Bridge']      = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
    dataset_1['Manhattan Bridge']     = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
    dataset_1['Queensboro Bridge']    = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
    dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
    dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
    dataset_1['Total']  = pandas.to_numeric(dataset_1['Total'].replace(',','', regex=True))
   
    dataset_test = dataset_1

    #Feature and target matrices
    X = dataset_test[['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge']]
#    X = dataset_test[['Brooklyn Bridge', 'Manhattan Bridge', 'Queensboro Bridge']]
#    X = dataset_test[['Brooklyn Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']]
#    X = dataset_test[['Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']]
    print (X)
    y = dataset_test[['Total']]
    
    X = X.to_numpy()
    y = y.to_numpy()
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)

    # Normalizing training and testing data
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    # Define the range of lambda to test
    lmbda = get_lambda_range()
    #lmbda = [1,100]
    MODEL = []
    MSE = []
    #temp2 = train_model(X_train, y_train, 1)
    #print("+++++++++++++")
    #print(error(X_test, y_test, temp2))
    y_predict_list = []
    print("+check point before train")
    for l in lmbda:
        # Train the regression model using a regularization parameter of l
        #model = train_model_lasso(X_train, y_train, l)
        model = train_model(X_train, y_train, l)
        
        # Evaluate the MSE on the test set
        mse = error(X_test, y_test, model)
        y_predict_list.append(model.predict(X_test))
        print("-check point of "+str(l))
        # Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)

    # Part 6
    # Plot the MSE as a function of lmbda
    plt.figure(1)
    plt.plot(lmbda, MSE, color="blue", linewidth=2.0)
    plt.title("MSE vs Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("MSE")
    print("+check point after plt")
    #plt.figure(2)
    #plt.plot(y_predict_list)
    #print(min(y_predict_list))
    # Part 7
    # Find best value of lmbda in terms of MSE
    ind =   MSE.index(min(MSE))
    [lmda_best, MSE_best, model_best] = [lmbda[ind], MSE[ind], MODEL[ind]]

    #evaluate the best model for
    #0.25 carat, 3 cut, 3 color, 5 clarity, 60 depth, 55 table, 4 x, 3 y, 2 z diamond (Use the Ridge regression model `train_model`)
    # NOTE: Make sure to normalize the given data


    X_for_pridiction_plot = normalize_test(X, trn_mean, trn_std)
    y_hat_for_pridiction_plot = model_best.predict(X_for_pridiction_plot)
    plt.figure(2)
    plt.scatter(range(len(y_hat_for_pridiction_plot)),y_hat_for_pridiction_plot, label="prediction", s=1)
    plt.scatter(range(len(y)),y, label="actual", s=1)
    plt.legend(fontsize=15, loc="upper left")    


    print(
        "Best lambda tested is "
        + str(lmda_best)
        + ", which yields an MSE of ----------------------------+"
        + str(MSE_best)
    )
    print ("Best model is"+str(model_best))
    
    


    return model_best


if __name__ == "__main__":
    model_best = main()
    # We use the following functions to obtain the model parameters instead of model_best.get_params()
    print(model_best.coef_)
    print(model_best.intercept_)
    plt.show()
