import streamlit as st
import pandas as pd
df1 = pd.read_csv("Datasets/Sales Volume.csv")

# Page Header and motive of supply side forecasting
st.markdown("<h1 style='text-align: center;'>Supply Side Forecasting</h1>", unsafe_allow_html=True)
st.subheader("The first step towards forecasting the supply, is to forecast demand.")
st.write("For Heritage, the first plan of action was to predict sales volume of milk for the next 4 quarters.")

#Sales Volume statistics, plots and predicitons

st.header("Sales Volume (in MLPD)")
st.subheader("The dateset for sales volume prediction: ")
st.write("You can find the logic for each column in the Heritage Foods presentation.")
df1 = df1.drop('Health Conciousness',axis = 1)
st.table(df1.head())
st.write("The dataset has a total of 12 entries.")

#Important plots for sales volume
st.subheader("Important plots for Sales Volume: ")
st.image("Images/Sales Volume vs Quarter.png",caption="Graph of Sales Volume vs Quarter. We can see peaks in the second quarter for each year, indicating seasonality ")
st.image("Images/Sales Volume Stationarity.png",caption='We can see stationarity in sales volume after taking a first difference')

#Model for Sales Volume
st.subheader("Model for Sales Volume: ")
st.write("Due to seasonality and stationarity, we decided to go with Seasonal ARIMA with exogenous(external) variables.")

st.markdown("**Criteria for model selection**: 80-20 training-test split and evaluting RMSE for test data. The minimum RMSE model was chosen.")
st.write('After using a spearman rank correlation matrix, a threshold of 0.72 was used to filter out relevant features:\n[Selling Value, Milk Procurement, Per Capita Income in Selling States, Prior CPI, Real PCI INR, Milk Prodution in Procurement States]')
st.write("After running a grid search over all possible external variable combinations and different orders for each combination, the best model was: ")
st.write("SARIMA(0,1,1)x(0,0,1,4) with the only exogenous variable of use being 'Selling value (per Lt in Rs)'. The order indicates that moving averages were a better indicator of forecast than auto regression.")
st.image("Images/Best Sales Volume Model.png",caption="Testing of the abovementioned model")

#Selling value: Idea, statisitcs, plots and models
st.header("Selling Value (per Lt in Rs)")
st.write("The next step for demand forecasting was to find the best model for prediciting Selling Value")
st.subheader("The dateset for sales volume prediction: ")
df2 = pd.read_csv("Datasets/Selling Value.csv")
df2 = df2.drop('Health Conciousness',axis = 1)
st.table(df2.head())
st.write('You may notice a difference from the sales volume database, i.e of the lagged variables.')
st.write('Lagged variables allow us to see the effect of previous values on latest values. For experimentation purposes, we manufactured some new variables with a lag of 1.')

#Plots for Selling Value
st.subheader('Important plots for selling value')
st.image('Images/Selling Value.png',caption = "Shows an increasing long term trend with no specific seasonality")
st.image('Images/Selling Value First Diff Stationarity.png',caption="Stationarity can be seen at the first difference")

#Model for selling value
st.subheader("Model for Selling Value: ")
st.write("Due presence of stationarity and no seasonality, this time the model to be used was ARIMA with exogenous variables (ARIMAX).")

st.markdown("**Criteria for model selection**: No test-training split this time, used all 12 datarows for training and calculated RMSE using model's prediction on exisiting data vs actual values. The minimum RMSE model was chosen.")
st.write('After using a spearman rank correlation matrix, a threshold of 0.7 was used to filter out relevant features:\n[Lagged Selling Value, Lagged Procurement Values, Procurement Value, Inflation (CPI), Per Capita Income in Selling States,Prior CPI, Lagged Milk Production, Milk Prodution in Procurement States]')
st.write("After running a grid search over all possible external variable combinations and different orders for each combination, the best model was: ")
st.write("ARIMA(0,1,0) with the exogenous variables of use being all the 8 variables above the threshold. The order indicates that the model is a random walk model, which means that immediate last value and external variables are the best regressors.")
st.image("Images/Selling Value Best Model.png",caption="Predicitions of the abovementioned model")
st.image("Images/Selling Value Predictions vs Actual.png",caption="Actual vs Selling Value, along with forecast")
st.write("The predictions stick very close to the actual values, depicting some sort of overfitting. This is the model of use as of now, until further experimentations.")

#How did we predict 4 regressors of selling value
st.header("Forecasting regressors for Selling Value")
st.subheader('To predict selling value, we need to forecast its regressors first: ')

#Procurement Values
st.subheader("1) Procurement Value")
st.write("We tried looking at an endogenous SARIMA Model for Procurement Values:")
st.image('Images/Procurement Values.png',caption='Augmented Dickey Fuller test says the series is stationary with the chance of it being a fluke at 5%')
st.write("We can also see some seasonality at the peaks.")
st.write("Model used was chosen using a training and test split of 80-20, the model with the least RMSE of the test split was used.")
st.write('Best Model: SARIMA(1,0,0)x(1,0,1,4). The autoregression parameter of ARIMA was chosen after looking at the PACF plot for the series.')
st.image('Images/Procurement Values PACF Plot.png',caption='PACF Plot')
st.image('Images/Actual vs Predicted with Forecast with m = 4.png',caption="Actual Vs Predicted values along with forecast")
st.write('The seasonal order was chosen after a grid search.')
df5 = pd.DataFrame({'Date':['2024-25 Q1','2024-25 Q2','2024-25 Q3','2024-25 Q4'],
                    'Procurement Value (per Lt in Rs)': ['43.3','43.3','41.8','42.2']})
st.write("Forecasted values:")
st.table(df5)

#Inflation
st.subheader("2) Inflation(CPI)")
st.write("A SARIMA(4,2,4)x(0,0,1,8) model was used here for forecasting inflation")
st.image('Images/CPI Second Difference Stationarity.png',caption='Tail end of CPI stationarity, achieved at a second difference')
st.image('Images/CPI ACF_PACF Second_Diff.png',caption='PACF and ACF plot for CPI, at difference = 2')
st.write('Due to strong spikes at lag 4 in both PACF and ACF, p and q were chosen to be 4. Seasonal order=(0,0,1,8) was chosen after experimentation.')
st.write("CPI model fit: ")
st.image('Images/CPI model fit.png',caption='Actual vs Predicted CPI along with forecasts')
df6 = pd.DataFrame({'Date':['2024-25 Q1','2024-25 Q2','2024-25 Q3','2024-25 Q4'],
                    'Infaltion (CPI)': ['153.14','155.06','158.19','159.28']})
st.write("Forecasted CPI:")
st.table(df6)
st.write("Note: First quarter CPI for 2024-25 was already published.")

#Milk Production in selling states
st.subheader("3) Milk Production in Procuring States")
st.write("Milk production for all 9(8) was done using ARIMA models, with each state's order being different.")
st.write("Milk Production Historic data: ")

df7 = pd.DataFrame( {
    'Year': list(range(2001, 2023)),
    'Punjab': [7932, 8173, 8391, 8554, 8909, 9168, 9282, 9387, 9389, 9423, 9551, 9724, 10011, 10351, 10774, 11282, 11855, 12599, 13348, 13394, 14077, 14301],
    'Rajasthan': [7758, 7789, 8054, 8310, 8713, 10309, 11377, 11931, 12330, 13234, 13512, 13946, 14573, 16934, 18500, 20850, 22427, 23668, 25573, 30723, 33265, 33307],
    'Uttar Pradesh': [14648, 15288, 15943, 16512, 17356, 18094, 18861, 19537, 20203, 21031, 22556, 23330, 24194, 25198, 26387, 27770, 29052, 30519, 31864, 31359, 33874, 36242],
    'Odisha': [929, 941, 997, 1283, 1342, 1431, 1625, 1598, 1651, 1671, 1721, 1724, 1861, 1903, 1930, 2003, 2088, 2311, 2370, 2373, 2402, 2476],
    'Telangana_Andhra': [5814,6584,6959,7257,7624,7938,8925,9570,10429,11203,12088,12762,13007,13863,15259,17399,18690,20460,20853,20479,21211,21258],
    'Tamil Nadu': [4988, 4622, 4752, 4784, 5474, 6277, 6540, 6651, 6787, 6831, 6968, 7005, 7049, 7132, 7244, 7556, 7742, 8362, 8759, 9790, 10107, 10317],
    'Karnataka': [4797, 4539, 3857, 3917, 4022, 4124, 4244, 4538, 4822, 5114, 5447, 5718, 5997, 6121, 6344, 6562, 7137, 7901, 9031, 10936, 11796, 12829],
    'Maharashtra': [6094, 6238, 6379, 6567, 6769, 6978, 7210, 7455, 7679, 8044, 8469, 8734, 9089, 9542, 10153, 10402, 11102, 11655, 12024, 13703, 14305, 15042]
})
st.table(df7.tail())
st.write("Note: Telangana and Andhra was merged due to missing values before 2015.")
st.write('Data from 2001 to 2022 was present. Data for 2023 and 2024 had to be predicted, but a rolling forecast was not used.')
st.image("Images/Punjab Milk Production Stationarity.png",caption="Example: Stationarity achieved at a differencing of 1 for Punjab. All states except UP (d=0) have the same d value of 1")
st.image("Images/Punjab Milk Production ACF_PACF Plot.png", caption="Example: Punjab's ACF-PACF plot")

st.write("Orders of each state: ")
df8 =  pd.DataFrame({
    'Punjab': [1, 1, 2],
    'Rajasthan': [1, 1, 2],
    'Uttar Pradesh': [1, 0, 2],
    'Odisha': [1, 1, 2],
    'Telangana_Andhra': [1, 1, 2],
    'Tamil Nadu': [1, 1, 1],
    'Karnataka': [1, 1, 2],
    'Maharashtra': [1, 1, 2]
})
st.table(df8)
st.write("Note: Rows are represented by p,d,q respectively")
st.write("Forecasted values: ")
df9 = pd.DataFrame({'Year':['2023-24','2024-25'],
        'Milk Production':[150934.027,154861.31]
    })

st.table(df9)
st.write("Note: Annual production was divided by 4 for Quarter to Quarter mapping.")

#Per Capita Income in Selling States
st.subheader("4) Per Capita Income")
st.write("To predict, per capita income in the selling states, Holt-Linear exponential smoothing was used, due to insignificant ACF-PACF plots of each state.")
st.image("Images/Delhi PCI ACF_PACF Plot.png",caption="Example: Delhi's ACF_PACF plot, other states followed the same pattern")
st.write("Exponential Smoothing allows you to predict future values, based on previous values with the help of an 'alpha' parameter that ranges from 0 to 1 and the higher it is, the higher the weightage for recent values.")
st.write("Holt Linear exponential smoothing allows us to use the properties of long term trend as well, something which is generally seen in per capita incomes.")
df10 = pd.DataFrame({
    "Year": [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "Delhi": [185001, 205568, 227900, 247209, 270261, 295558, 318323, 338730, 355798, 322311, 376217, 430120, 461910],
    "Haryana": [106085, 121269, 137770, 147382, 164963, 184982, 208437, 223022, 232530, 224587, 264729, 296592, 325759],
    "Uttarakhand": [100314, 113654, 126356, 136099, 147936, 161752, 180858, 186207, 190558, 174526, 205246, 230994, 260201],
    "Uttar Pradesh": [32002, 35812, 40124, 42267, 47118, 52671, 57944, 62350, 65660, 61809, 73841, 83636, 93514],
    "Odisha": [48387, 54762, 60687, 63345, 64835, 77507, 87055, 98005, 104633, 103203, 126437, 145202, 161437],
    "Telangana": [91121, 101007, 112162, 124104, 140840, 159395, 179358, 209848, 231326, 225734, 269161, 311649, 347299],
    "Andhra Pradesh": [69000, 74687, 82870, 93903, 108002, 120676, 138299, 154031, 160341, 168063, 197214, 219881, 242479],
    "Tamil Nadu": [93112, 105340, 116960, 129494, 142028, 156595, 175276, 194373, 206165, 209628, 242253, 275583, 313955],
    "Kerala": [97912, 110314, 123388, 135537, 148133, 166246, 183252, 205437, 208879, 194432, 234435, 263945, 263945],
    "Karnataka": [90263, 102319, 118829, 130024, 148108, 169898, 185840, 205245, 222141, 221781, 266866, 304474, 332926],
    "Maharashtra": [99597, 112092, 125261, 132836, 146815, 163726, 172663, 182865, 189889, 183704, 215233, 218970, 229431]
})
st.table(df10.tail())
st.write("Values were present from 2011 to 2023. A few NaN were replaced using exponential smoothing.")

st.write("Results: ")
df11 = {
    "State": [
        "Delhi", "Haryana", "Uttarakhand", "Uttar Pradesh", "Odisha", 
        "Telangana", "Andhra Pradesh", "Tamil Nadu", "Kerala", "Karnataka", "Maharashtra"
    ],
    "Forecasted PCI (2024-25) in INR": [
        479710.78, 338873.97, 271702.24, 99200.52, 175527.06, 
        378547.62, 264323.63, 349373.42, 272664.90, 360092.54, 237779.97
    ]
}
st.table(df11)
st.markdown(f"Average PCI came out be at INR **293,436**.")
st.write("Note: Annual PCI was divided by 4 for Quarter to Quarter mapping.")


#Selling Value Predictions
st.header("Forecasting Selling Value")
st.write("Now that the best model for selling value has been achieved, we need to do the predictions.")
st.write("Predictions were done using a rolling forecast, i.e, once a single forecast is achieved, we put it back in the model for training to get the next forecast.")
st.write("Using the entire dataset, ARIMA(1,1,0) and regressors: \n [Lagged Selling Value, Lagged Procurement Values, Procurement Value, Inflation (CPI), Per Capita Income in Selling States,Prior CPI, Lagged Milk Production, Milk Prodution in Procurement States], we were able to predict the following values: ")
df3 = pd.DataFrame({'Date':['2024-25 Q1','2024-25 Q2','2024-25 Q3','2024-25 Q4'],
                    'Selling Value (per Lt in Rs)': ['56.17','55.7','56.68','55.81']})
st.table(df3)

#Sale Volume Predictions
st.header("Forecasting Sales Volume")
st.write("Now that selling value has been predicted, we use it to predict sales volume, once again with a rolling forecast.")
st.write("Using the entire dataset, SARIMA(0,1,1)x(0,0,1,4) with 'Selling Value (per Lt in Rs) as a regressor, we were able to predict the following values: ")
df4 = pd.DataFrame({'Date':['2024-25 Q1','2024-25 Q2','2024-25 Q3','2024-25 Q4'],
                    'Selling Value (per Lt in Rs)': ['56.17','55.7','56.68','55.81'],
                    'Sales Volume (in MLPD)': ['1.1151','1.121864','1.13333','1.125221']})
st.table(df4)
