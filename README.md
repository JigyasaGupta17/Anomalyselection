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
