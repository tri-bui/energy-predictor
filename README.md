# Great Energy Predictor

## Background
In a broad sense, a lot of buildings today have a problem of unnecessary or inefficient energy usage, which is both wasteful for the owner who has to pay for it and detrimental to the environment because of the excessive carbon emissions. Retrofitting these buildings to make them more energy-efficient requires significant investments, but doing so could be beneficial in the long term for the investors, the building owners, and the environment. This is done through something called pay-for-performance financing, which bases payments on the energy savings made possible by the retrofit. But the problem here is that there is no way of knowing how much energy a building is saving from the retrofit. 

The solution is to build a model to estimate the energy usage of buildings if the retrofits had not been made. These estimates are used to compare to the actual energy usage after the retrofit has been made to yield the energy savings, which the payments are based on. Accurate estimates could potentially draw more investors and lower the cost of financing for building owners. 

## Data
The [data](https://www.kaggle.com/c/ashrae-energy-prediction) being used is a collection of 1,449 buildings in 16 sites around the world. It contains measurements from 4 types of energy meters (electricity, chilled water, steam, and hot water) over a period of an entire year (2016). Each observation is a meter reading from a certain meter in a certain building at a certain point in time. This means that at any point in time, there could be up to 4 observations pertaining to the same building, but not all recorded buildings have all 4 meters. Also included are several details about each building and weather data from the closest weather station.

