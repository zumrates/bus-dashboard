import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Otobüs Bakım Önceliklendirme Dashboardu", layout="wide")

# Başlık
st.title("Otobüs Bakım Önceliklendirme Dashboardu")

# Veri yükleme
sensor_data = pd.read_csv('demo_sensor_data.csv')
yolcu_data = pd.read_csv('iett_hat_yogunluk.csv')
kaza_data = pd.read_csv('kaza_data.csv')
bozuk_data = pd.read_csv('bozuk_satih_data.csv')

# Özellikler ve etiket
X = sensor_data.drop(columns=['label', 'OtobusID'])
y = sensor_data['label']

# Model eğitimi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Tahminler ve varyans (10 kez tekrar için ensemble örneği)
preds_list = []
for seed in range(10):
    temp_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=seed)
    temp_model.fit(X_train, y_train)
    preds_list.append(temp_model.predict_proba(X)[:,1])

ensemble_preds = np.mean(preds_list, axis=0)
variance = np.var(preds_list, axis=0)

sensor_data['ML_Prediction'] = (ensemble_preds >= 0.5).astype(int)
sensor_data['Variance'] = variance

# Maksimum yolcu ve sefer sayısı (normalize için)
max_yolcu = yolcu_data['YOLCU_SAYISI'].max()
max_sefer = 500  # örnek, gerçek veriyle güncelle

# Öncelik skoru hesaplama
def calculate_priority(ml_pred, yolcu_sayisi, sefer_sayisi, has_event, variance):
    base_score = 0.5 * (yolcu_sayisi / max_yolcu) + 0.3 * (sefer_sayisi / max_sefer) + 0.2 * has_event
    priority = ml_pred * base_score * (1 - variance)
    return priority * 100

# Sonuç hesaplama
priority_scores = []
for idx, row in sensor_data.iterrows():
    hat_kodu = np.random.choice(yolcu_data['SHATKODU'])
    yolcu_sayisi = yolcu_data[yolcu_data['SHATKODU'] == hat_kodu]['YOLCU_SAYISI'].values[0]
    sefer_sayisi = np.random.randint(100, 500)  # örnek
    has_event = int(
        (kaza_data['NBOYLAM'].between(28.9, 29.2).sum() > 0) or
        (bozuk_data['NBOYLAM'].between(28.9, 29.2).sum() > 0)
    )
    priority = calculate_priority(row['ML_Prediction'], yolcu_sayisi, sefer_sayisi, has_event, row['Variance'])
    priority_scores.append(priority)

sensor_data['Priority_Score'] = priority_scores

# Dashboard
top_n = st.slider('Kaç Otobüsü Gösterelim?', 5, 20, 10)
top_priority = sensor_data.sort_values('Priority_Score', ascending=False).head(top_n)

st.subheader("Öncelikli Otobüsler")
st.dataframe(top_priority[['OtobusID', 'ML_Prediction', 'Variance', 'Priority_Score']])

if st.button("Seçili Otobüse Bildirim Gönder"):
    st.success("Bildirim gönderildi!")

