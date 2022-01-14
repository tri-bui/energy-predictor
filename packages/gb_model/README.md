# Gradient Boosting Regression Model

This package preprocesses input meter and weather data, engineers and selects features, and makes hourly energy consumption predictions using a trained gradient boosting model. The trained model is either a LightGBM or an XGBoost model. The model was trained on the data of 1,449 unique buildings, each of which have between 1 and 4 different meter types (from electricity, chilled water, steam, and hot water).

As the study of power consumption in buildings is a very specific problem, this application was designed to make predictions for these 1,449 buildings ONLY. Any new buildings would require additional training data for those buildings and the model would have to be retrained and updated.