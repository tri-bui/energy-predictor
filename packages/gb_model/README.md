# Gradient Boosting Regression Model

This package preprocesses input meter and weather data, engineers and selects features, and makes hourly energy consumption predictions using a trained gradient boosting model. The trained model is either a LightGBM or an XGBoost model. The model was trained on the data of 1,449 unique buildings, each of which have between 1 and 4 different meter types (from electricity, chilled water, steam, and hot water).

As the study of power consumption in buildings is a very specific problem, this application was designed to make predictions for these 1,449 buildings ONLY. Any new buildings would require additional training data for those buildings and the model would have to be retrained and updated.

## Prediction Pipeline

1. Read in building, meter, and weather data
2. Validate meter and weather data and convert timestamps to datetime
3. Preprocess weather data
    a. Convert timestamps from UTC to local time
    b. Reindex data to include a timestamp for every hour within the time interval at every site
    c. Impute missing values using cubic and linear interpolation, followed by forward and backward fills (this step only applies to sites that are not missing 100% of their values in the feature being filled)
    d. Copy data from one site's feature to another site (this step applies to sites missing 100% of their values in the feature)
4. Merge all input data
5. Engineer and select features
    a. Create new features for relative humidity and wind direction cartesian components
    b. Extract time components from timestamps
    c. Create new features for country and holiday indicator
    d. Select features to match the data the model was trained on
6. Check for missing values
7. Make predictions
    a. Split data by meter type
    b. For each meter type:
        i. Encode rare categorical labels
        ii. Encode categorical features using the label's target mean
        iii. Scale features using the mean and standard deviation
        iv. Load model and make predictions
        v. Inverse-transform predictions
    c. Combine predictions from all meter types
8. Log the model version, input meter and weather data, and predictions
9. Convert predictions and model version into JSON format