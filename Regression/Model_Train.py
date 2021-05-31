import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoLars
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import _pickle as cPickle
import numpy as np

def Read_Data(Data_Path):
    Data = pd.read_csv(Data_Path)
    return Data

def Label_Encoding(Data, Column_Name):
    le = LabelEncoder()
    Data[Column_Name] = le.fit_transform(Data[Column_Name])
    return Data

def one_hot_encoding(Data, Column_Name):
    one_hot = pd.get_dummies(Data[Column_Name])
    Data = Data.drop(Column_Name, axis=1)
    Data = Data.join(one_hot)
    return Data

def Fill_Null_Mode(Data, Column_Name):
    Data[Column_Name] = Data[Column_Name].fillna(Data[Column_Name].mode()[0])
    return Data
def Fill_Null_Mean(Data, Column_Name):
    Data[Column_Name] = Data[Column_Name].fillna(Data[Column_Name].mean())
    return Data
def Sum_Two_Columns(Data, New_Column, Column_Name1, Column_Name2):
    Data[New_Column] = Data[Column_Name1] + Data[Column_Name2]
    Data = Data.drop(Column_Name1, axis=1)
    Data = Data.drop(Column_Name2, axis=1)
    return Data

def PreProccessing(Data):

    Data = Data.drop("CarName", axis=1)

    # Every idi fuel system is diesel and vice versa then label encoding
    Data.loc[(Data['fueltype'].isnull()) & (Data["fuelsystem"] == "idi"), 'fueltype'] = "diesel"
    Data.loc[(Data['fueltype'].isnull()) & (Data["fuelsystem"] != "idi"), 'fueltype'] = "gas"
    Data = Label_Encoding(Data, 'fueltype')

    # Get most used Value in symboling to fill nulls then label encoding
    Data = Fill_Null_Mode(Data, "symboling")
    Data = Label_Encoding(Data, "symboling")

    # Get most used Value in aspiration to fill nulls then label encoding
    Data = Fill_Null_Mode(Data, "aspiration")
    Data = Label_Encoding(Data, "aspiration")

    # Get most used Value in doornumber to fill nulls then label encoding
    Data = Fill_Null_Mode(Data, "doornumber")
    Data = one_hot_encoding(Data, "doornumber")

    # Get most used Value in carbody to fill nulls then label encoding
    Data = Fill_Null_Mode(Data, "carbody")
    Data = Label_Encoding(Data, "carbody")

    # Get most used Value in drivewheel to fill nulls then label encoding
    Data = Fill_Null_Mode(Data, "drivewheel")
    Data = Label_Encoding(Data, "drivewheel")

    # Get most used Value in enginelocation to fill nulls then label encoding
    Data = Fill_Null_Mode(Data, "enginelocation")
    Data = Label_Encoding(Data, "enginelocation")

    # Get mean of wheelbase to fill nulls
    Data = Fill_Null_Mean(Data, "wheelbase")

    # Get mean of carlength and carwidth to fill nulls
    Data = Fill_Null_Mean(Data, "carlength")
    Data = Fill_Null_Mean(Data, "carwidth")

    # Since correlation between carlength, carwidth is high we can add them to cardimensions
    Data = Sum_Two_Columns(Data, "cardimensions", "carlength", "carwidth")

    # Get mean of carheight to fill nulls
    Data = Fill_Null_Mean(Data, "carheight")

    # Get mean of curbweight to fill nulls
    Data = Fill_Null_Mean(Data, "curbweight")

    # Get most used Value in enginetype to fill nulls then label encoding
    Data = Fill_Null_Mode(Data, "enginetype")
    Data = Label_Encoding(Data, "enginetype")

    #Replace enginesize according to cylindernumber
    Data.loc[(Data['enginesize'].isnull()) & (Data["cylindernumber"] == "three"), 'enginesize'] = 45
    Data.loc[(Data['enginesize'].isnull()) & (Data["cylindernumber"] == "four"), 'enginesize'] = 113
    Data.loc[(Data['enginesize'].isnull()) & (Data["cylindernumber"] == "five"), 'enginesize'] = 170
    Data.loc[(Data['enginesize'].isnull()) & (Data["cylindernumber"] == "six"), 'enginesize'] = 220
    Data.loc[(Data['enginesize'].isnull()) & (Data["cylindernumber"] == "eight"), 'enginesize'] = 283
    Data.loc[(Data['enginesize'].isnull()) & (Data["cylindernumber"] == "twelve"), 'enginesize'] = 300
    Data.loc[(Data['enginesize'].isnull()) & (Data["cylindernumber"].isnull()), 'enginesize'] = Data["enginesize"].mean()
    #And now do the opposite
    Data.loc[(Data['cylindernumber'].isnull()) & (Data["enginesize"] <= 70), 'cylindernumber'] = "three"
    Data.loc[(Data['cylindernumber'].isnull()) & (Data["enginesize"] <= 156) & (
                Data["enginesize"] > 70), 'cylindernumber'] = "four"
    Data.loc[(Data['cylindernumber'].isnull()) & (Data["enginesize"] <= 183) & (
                Data["enginesize"] > 156), 'cylindernumber'] = "five"
    Data.loc[(Data['cylindernumber'].isnull()) & (Data["enginesize"] <= 258) & (
                Data["enginesize"] > 183), 'cylindernumber'] = "six"
    Data.loc[(Data['cylindernumber'].isnull()) & (Data["enginesize"] <= 308) & (
                Data["enginesize"] > 258), 'cylindernumber'] = "eight"
    Data.loc[(Data['cylindernumber'].isnull()) & (Data["enginesize"] <= 1000) & (
                Data["enginesize"] > 308), 'cylindernumber'] = "twelve"
    Data.loc[(Data['cylindernumber'].isnull()) & (Data["enginesize"].isnull()), 'cylindernumber'] = "three"

    # Label encoding of cylindernumber
    Data = Label_Encoding(Data, "cylindernumber")

    # Get most used Value in fuelsystem to fill nulls then label encoding
    Data = Fill_Null_Mode(Data, "fuelsystem")
    Data = Label_Encoding(Data, "fuelsystem")

    # Get mean of boreratio to fill nulls
    Data = Fill_Null_Mean(Data, "boreratio")

    # Get mean of stroke to fill nulls
    Data = Fill_Null_Mean(Data, "stroke")

    # Get mean of compressionratio to fill nulls
    Data = Fill_Null_Mean(Data, "compressionratio")

    # Get mean of horsepower to fill nulls
    Data = Fill_Null_Mean(Data, "horsepower")

    # Get mean of horsepower to fill nulls
    Data = Fill_Null_Mean(Data, "peakrpm")

    # A single variable mileage can be calculated
    Data = Sum_Two_Columns(Data, "mileage", "citympg", "highwaympg")
    # Get mean of mileage to fill nulls
    Data = Fill_Null_Mean(Data, "mileage")

    return Data

def Save_Model(Model, File_Name):
    f = open(File_Name, "wb")
    f.write(cPickle.dumps(Model))
    f.close()

def Train_model(Data, Regression_Technique,scaler):
    # Split data 80%-20%
    Train_Data, Test_Data = train_test_split(Data, train_size=0.8, test_size=0.2, random_state=100)
    num_vars = ['wheelbase', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',
                'horsepower', 'peakrpm', 'cardimensions', 'mileage']
    Train_Data[num_vars] = scaler.fit_transform(Train_Data[num_vars])
    y_train = Train_Data.pop('price')
    X_train = Train_Data

    start = time.time()
    if Regression_Technique == "linear Regression":
        Model = LinearRegression()
        Model.fit(X_train, y_train)
        Save_Model(Model, "Linear_Regression_Model.cpickle")
    elif Regression_Technique == "Ridge Regression":
        Model = Ridge()
        Model.fit(X_train, y_train)
        Save_Model(Model, "Ridge_Regression_Model.cpickle")
    elif Regression_Technique == "LarsLasso Regression":
        Model = LassoLars()
        Model.fit(X_train, y_train)
        Save_Model(Model, "LarsLasso_Regression_Model.cpickle")


    stop = time.time()
    Train_Time = stop - start
    num_vars = ['wheelbase', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',
                'horsepower', 'peakrpm', 'cardimensions', 'mileage']
    Test_Data[num_vars] = scaler.transform(Test_Data[num_vars])
    y_test = Test_Data.pop('price')
    X_test = Test_Data
    y_pred = Model.predict(X_test)
    R2_score = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)

    return Train_Time, R2_score, MSE

def Test_Model(scaler):
    Test_Path = "CarPrice_testing.csv"
    Test_Data = pd.read_csv(Test_Path)
    Cleaned_Test_Data = PreProccessing(Test_Data)
    num_vars = ['wheelbase', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',
                'horsepower', 'peakrpm', 'cardimensions', 'mileage']
    Cleaned_Test_Data[num_vars] = scaler.fit_transform(Cleaned_Test_Data[num_vars])
    Actual_Predicted_Test = Cleaned_Test_Data.pop("price")

    Model_Name = 'Ridge_Regression_Model.cpickle'
    Model = cPickle.loads(open(Model_Name, "rb").read())
    Test_Predictions = Model.predict(Cleaned_Test_Data)
    Test_Predictions = np.abs(Test_Predictions)
    R2_score = r2_score(Actual_Predicted_Test, Test_Predictions)
    MSE = mean_squared_error(Actual_Predicted_Test, Test_Predictions)
    print(Test_Predictions)
    print("R2_score: " + str(R2_score))
    print("MSE Value: " + str(MSE))


if __name__ == '__main__':
    File_Path = "CarPrice_training.csv"
    Data = Read_Data(File_Path)
    Cleaned_Data = PreProccessing(Data)
    Regression_Technique = "Ridge Regression"
    scaler = MinMaxScaler()
    Train_Time, R2_score, MSE = Train_model(Cleaned_Data, Regression_Technique,  scaler)
    Test_Model(scaler)
    #print("Train Time: " + str(Train_Time))
    #print("R2_score: " + str(R2_score))
    #print("MSE Value: " + str(MSE))
