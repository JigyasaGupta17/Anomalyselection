# 📈 Google Trends Anomaly Detector

This project explores spikes in public search interest using **Google Trends**, **Python**, and **Machine Learning**.

Using keywords like `"iPhone"` and `"Samsung"`, I built an anomaly detection model to identify unusual search patterns — and connected these patterns to real-world marketing triggers like product launches, sales, or possibly even inorganic activity.

---

## 🔍 What I Did
- Pulled 30 days of Google Trends data using `pytrends`
- Applied `Isolation Forest` for anomaly detection
- Visualized the anomalies using `matplotlib` and `seaborn`
- Interpreted spikes in context of both organic and suspicious behavior

---

## 💡 Real-World Context

In the advertising world, not all traffic spikes are real. This kind of anomaly detection is key to:
- Spotting **ad fraud**
- Understanding **consumer behavior**
- Improving **media strategy** with clean data

This logic is inspired by real-time fraud filtering practices at my workplace, **mFilterIt** — where we detect and mitigate invalid traffic at scale.

---

## 📊 Output

![Anomaly Chart](images/anomaly_chart.png)

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
from pytrends.request import TrendReq
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import seaborn as sns

# Initialize PyTrends
pytrends = TrendReq(hl='en-IN', tz=330)

# Keywords to analyze
kw_list = ["iPhone", "Samsung"]

# === PART 1: Trend data for past 30 days ===
pytrends.build_payload(kw_list, cat=0, timeframe='today 1-m', geo='IN')  # All-India

# Fetch data
data = pytrends.interest_over_time().reset_index().drop(columns=['isPartial'])

# === PART 2: Apply Isolation Forest per keyword ===
for kw in kw_list:
    model = IsolationForest(contamination=0.1, random_state=42)
    data[f'{kw}_anomaly'] = model.fit_predict(data[[kw]])

# === PART 3: Plotting anomalies ===
plt.figure(figsize=(14, 6))
for kw in kw_list:
    plt.plot(data['date'], data[kw], label=kw)
    anomalies = data[data[f'{kw}_anomaly'] == -1]
    plt.scatter(anomalies['date'], anomalies[kw], s=100, label=f'{kw} Anomaly')

plt.title("Search Interest with Anomaly Detection: iPhone vs Samsung (India, Last 30 Days)")
plt.xlabel("Date")
plt.ylabel("Interest")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# === PART 4: Geo-Level Breakdown ===
# Let's fetch top cities for each keyword
for kw in kw_list:
    print(f"\n📍 Top Regions for: {kw}")
    geo_df = pytrends.interest_by_region(resolution='CITY', inc_low_vol=True, inc_geo_code=False)
    print(geo_df[kw].sort_values(ascending=False).head(10))

# === PART 5: Seasonal Comparison (July vs October) ===
# You can loop over historical timeframes manually for year-on-year comparison

def get_monthly_trend(keyword, month_range, year):
    timeframe = f"{year}-{month_range[0]}-01 {year}-{month_range[1]}-31"
    pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='IN')
    return pytrends.interest_over_time()[keyword].reset_index()

# Example: iPhone - July vs October
july_data = get_monthly_trend("iPhone", ("07", "07"), "2023")
oct_data = get_monthly_trend("iPhone", ("10", "10"), "2023")

plt.figure(figsize=(12, 5))
plt.plot(july_data['date'], july_data['iPhone'], label='July 2023')
plt.plot(oct_data['date'], oct_data['iPhone'], label='October 2023')
plt.title("Seasonal Trend Comparison for 'iPhone' (July vs October 2023)")
plt.xticks(rotation=45)
plt.ylabel("Search Interest")
plt.legend()
plt.tight_layout()
plt.show()

