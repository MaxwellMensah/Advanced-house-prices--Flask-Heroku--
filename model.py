import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv("TrainClean.csv")

# le = LabelEncoder()
# for ExterQual in df:
#     df[ExterQual] = le.fit_transform(df[ExterQual])

#       10    Very Excellent
#        9	Excellent
#        8	Very Good
#        7	Good
#        6	Above Average
#        5	Average
#        4	Below Average
#        3	Fair
#        2	Poor
       
       
# df['OverallQual'] = df['OverallQual'].map({1:Very Poor}, {2:Fair}, {3:})

# for GarageFinish in df:
#     df[GarageFinish] = le.fit_transform(df[GarageFinish])

features = df[['OverallQual', 'GrLivArea', 'GarageCars', 'KitchenQual', 'ExterQual'
               , 'SalePrice']]

X = features.iloc[:, :5]
y = features.iloc[:, -1]
regressor = GradientBoostingRegressor(max_depth=100, max_features='sqrt',
                          min_samples_leaf=4, min_samples_split=5,
                          n_estimators=200)  

# Fitting the model to the training data...
regressor.fit(X, y)

# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(regressor, open('model1.pkl','wb'))



'''#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2.6, 8, 10.1]]))''' 