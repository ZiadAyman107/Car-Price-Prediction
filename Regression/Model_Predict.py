import _pickle as cPickle
import pandas as pd
from Model_Train import PreProccessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error



Test_Path = "CarPrice_testing.csv"
Test_Data = pd.read_csv(Test_Path)
Cleaned_Test_Data = PreProccessing(Test_Data)
scaler = MinMaxScaler()
num_vars = ['wheelbase', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',
                'horsepower', 'peakrpm', 'cardimensions', 'mileage']
Cleaned_Test_Data[num_vars] = scaler.fit_transform(Cleaned_Test_Data[num_vars])
Actual_Predicted_Test = Cleaned_Test_Data.pop("price")

Model_Name = 'Linear_Regression_Model.cpickle'
Model = cPickle.loads(open(Model_Name, "rb").read())
Test_Predictions = Model.predict(Cleaned_Test_Data)

R2_score = r2_score(Actual_Predicted_Test, Test_Predictions)
MSE = mean_squared_error(Actual_Predicted_Test, Test_Predictions)
print(Test_Predictions)
print("R2_score: " + str(R2_score))
print("MSE Value: " + str(MSE))