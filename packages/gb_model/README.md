# Gradient Boosting Regression Model

This package preprocesses input meter and weather data, engineers and selects features, and makes hourly energy consumption predictions using a trained gradient boosting model. The trained model is either a LightGBM or an XGBoost model. The model was trained on the data of 1,449 unique buildings, each of which have between 1 and 4 different meter types (from electricity, chilled water, steam, and hot water).

As the study of power consumption in buildings is a very specific problem, this application was designed to make predictions for these 1,449 buildings ONLY. Any new buildings would require additional training data for those buildings and the model would have to be retrained and updated.

## Prediction Pipeline

<ol>
    <li>Read in building, meter, and weather data</li>
    <li>Validate meter and weather data and convert timestamps to datetime</li>
    <li>Preprocess weather data<ol>
        <li type='a'>Convert timestamps from UTC to local time</li>
        <li type='a'>Reindex data to include a timestamp for every hour within the time interval at every site</li>
        <li type='a'>Impute missing values using cubic and linear interpolation, followed by forward and backward fills (this step only applies to sites that are not missing 100% of their values in the feature being filled)</li>
        <li type='a'>Copy data from one site's feature to another site (this step applies to sites missing 100% of their values in the feature)</li>
    </ol></li>
    <li>Merge all input data</li>
    <li>Engineer and select features<ol>
        <li type='a'>Create new features for relative humidity and wind direction cartesian components</li>
        <li type='a'>Extract time components from timestamps</li>
        <li type='a'>Create new features for country and holiday indicator</li>
        <li type='a'>Select features to match the data the model was trained on</li>
    </ol></li>
    <li>Check for missing values</li>
    <li>Make predictions<ol>
        <li type='a'>Split data by meter type</li>
        <li type='a'>For each meter type:<ol>
            <li type='i'>Encode rare categorical labels</li>
            <li type='i'>Encode categorical features using the label's target mean</li>
            <li type='i'>Scale features using the mean and standard deviation</li>
            <li type='i'>Load model and make predictions</li>
            <li type='i'>Inverse-transform predictions</li>
        </ol></li>
        <li type='a'>Combine predictions from all meter types</li>
    </ol></li>
    <li>Log the model version, input meter and weather data, and predictions</li>
    <li>Convert predictions and model version into JSON format</li>
</ol>