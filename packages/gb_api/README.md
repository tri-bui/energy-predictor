# Great Energy Predictor

## Web Service documentation

This application predicts the power consumption of 1,449 different buildings based on 4 different meter types: electricity, chilled water, steam, and hot water. Not all buildings have every meter. Predictions are made at hourly points in time (example: Jan 1, 2018 12:00:00) based on the timestamps found in the provided meter data. This also requires that weather data be provided. If the predictions are for a future point in time, the weather forecast data would suffice. Data requirements are described in the NOTE below.

As the study of power consumption in buildings is a very specific problem, this application was designed to make predictions for these 1,449 buildings ONLY. Any new buildings would require additional training data for those buildings and the model would have to be retrained and updated.

### Endpoints

#### /health
- Request: "GET"
- Returns: "ok" for 200 status
- Format: String

#### /version
- Request: "GET"
- Returns: REST API version and model package version
- Format: JSON

#### /v1/predict
- Request: "POST"
- Body: Meter and weather data
- Returns: Energy consumption predictions in kWh and model version
- Format: JSON

#### NOTE: Making a POST request to the prediction endpoint requires BOTH the meter and weather data in the body. This can either be sent as raw JSON text or as JSON files labeled with keys:
- Format: {"meter": {METER DATA}, "weather": {WEATHER DATA}}
- Meter data must have the following columns: ["timestamp", "building_id", "meter"]
- Weather data must have the following columns: ["timestamp", "site_id", "air_temperature", "dew_temperature", "sea_level_pressure", "wind_speed", "wind_direction"]