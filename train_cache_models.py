# train_cache_models.py
import pandas as pd
import os
from prophet import Prophet
import joblib
from tqdm import tqdm

DATA_PATH = "Vegetable and Fruits Prices  in India.xlsx"
MODEL_DIR = "models_cache"
os.makedirs(MODEL_DIR, exist_ok=True)

xl = pd.ExcelFile(DATA_PATH)
df = xl.parse(xl.sheet_names[0])
# normalize columns similar to app.py
date_col = [c for c in df.columns if "date" in c.lower()][0]
item_col = [c for c in df.columns if "item" in c.lower() or "name" in c.lower()][0]
price_col = [c for c in df.columns if "price" in c.lower() or "amount" in c.lower()][0]
df = df[[date_col, item_col, price_col]].copy()
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
df = df.dropna(subset=[date_col, item_col, price_col])
df = df.rename(columns={date_col:'Date', item_col:'Item', price_col:'Price'})

items = df['Item'].unique()
for item in tqdm(items):
    sub = df[df['Item']==item].set_index('Date').resample('D')['Price'].mean().reset_index()
    sub = sub.rename(columns={'Date':'ds','Price':'y'}).dropna()
    if len(sub) < 30:
        continue
    m = Prophet(yearly_seasonality=True)
    try:
        m.fit(sub)
        joblib.dump(m, os.path.join(MODEL_DIR, f"{item.replace('/','_')}_prophet.joblib"))
    except Exception as e:
        print("Failed for", item, e)
