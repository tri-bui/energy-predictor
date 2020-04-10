# Great Energy Predictor

### Background
In a broad sense, a lot of buildings today have a problem of unnecessary or inefficient energy usage, which is both wasteful for the owner who has to pay for it and detrimental to the environment because of the carbon emissions. Retrofitting these buildings to make them more energy-efficient requires significant investments, but doing so could be beneficial in the long term for the investors, the building owners, and the environment. This is done through something called pay-for-performance financing, which bases payments on the energy savings made possible by the retrofit. But the issue here is that these energy savings are calculated based on estimates made by suboptimal models.

### Summary
The solution is to build a scalable model to make better estimations than current estimation methods. This model will estimate the energy usage of buildings if the retrofits had not been made. These estimates are used to compare to the actual energy usage after the retrofit has been made to yield the energy savings, which the payments are based on. Better estimates could potentially draw more investors and lower the cost of financing for building owners. 

### Specifications
The final deliverable will be a prediction application deployed as a web service. This project will require a 4-core CPU and 16 GB of RAM. No GPU is required.

### Data
The data being used is a collection of 1,449 buildings in 16 sites around the world. It contains measurements from 4 types of energy meters (electricity, chilled water, steam, and hot water) over a period of an entire year (2016). Each observation is a meter reading from a certain meter in a certain building at a certain point in time. This means that at any point in time, there could be up to 4 observations pertaining to the same building, but not all recorded building have all 4 meters. Also included are several details about each building and weather data from the closest weather station.

### Source
https://www.kaggle.com/c/ashrae-energy-prediction/data