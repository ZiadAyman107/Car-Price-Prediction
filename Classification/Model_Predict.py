import _pickle as cPickle
import pandas as pd
from Model_Train import PreProccessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score



Test_Path = "CarPrice_testing_classification.csv"
Test_Data = pd.read_csv(Test_Path)
Cleaned_Test_Data = PreProccessing(Test_Data)
scaler = MinMaxScaler()
num_vars = ['wheelbase', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio',
                'horsepower', 'peakrpm', 'cardimensions', 'mileage']
Cleaned_Test_Data[num_vars] = scaler.fit_transform(Cleaned_Test_Data[num_vars])
Actual_Predicted_Test = Cleaned_Test_Data.pop("Category")

Model_Name = 'SVM_Model.cpickle'
Model = cPickle.loads(open(Model_Name, "rb").read())
Test_Predictions = Model.predict(Cleaned_Test_Data)


Modified_Predictions = []
for i in Test_Predictions:
    if i == 0:
        Modified_Predictions.append("High")
    else:
        Modified_Predictions.append("Low")

R2_score = r2_score(Actual_Predicted_Test, Test_Predictions)
Accuracy = accuracy_score(Actual_Predicted_Test, Test_Predictions)
print(Modified_Predictions)
print("R2_score: " + str(R2_score))
print("Accuracy: " + str(100 * Accuracy))

