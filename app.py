from prophet import Prophet
import pandas as pd
import holidays
import os

# Set path and load data
os.chdir("C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk")
data = pd.read_csv("airtel.csv")

# Convert and rename columns for Prophet
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
data = data.rename(columns={"Date": "ds", "Price": "y"})

# Add Indian holidays
years = pd.DatetimeIndex(data["ds"]).year.unique()
ind_holidays = holidays.India(years=years)
holiday_df = pd.DataFrame({"ds": list(ind_holidays.keys()), "holiday": "india_national"})

# Fit Prophet model
model = Prophet(holidays=holiday_df)
model.fit(data)

# Forecast 30 days ahead
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Save last 30 days of forecast
forecast[["ds", "yhat"]].tail(30).to_csv("forecast_prophet.csv", index=False)
