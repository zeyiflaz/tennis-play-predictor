# 🎯 Hava Durumuna Göre Oyun Tahmini - Makine Öğrenmesi Projesi

Bu projede, hava durumu verileri kullanılarak **oyun oynanır mı?** kararını ve **sıcaklık değerini** tahmin etmek amacıyla makine öğrenmesi teknikleri uygulanmıştır.

## 🔍 Proje Amacı

Veri seti; `outlook`, `temperature`, `humidity`, `windy` ve `play` gibi sütunlar içermektedir. Projede:

- **Veri ön işleme (LabelEncoder ve OneHotEncoder)**
- **Veri birleştirme ve dönüşüm**
- **Linear Regression ile tahmin**
- **Statsmodels ile istatistiksel analiz**
- **R², p-value gibi değerlere göre model yorumu**

adımları gerçekleştirilmiştir.

## 📁 Kullanılan Veri Seti

Veri dosyası: `odev_tenis.csv`

Sütunlar:
- `outlook`: Hava durumu (sunny, overcast, rainy)
- `temperature`: Sıcaklık değeri (sayısal)
- `humidity`: Nem oranı
- `windy`: Rüzgarlı mı (True / False)
- `play`: Oyun oynanır mı? (Yes / No)

## 🧪 Kullanılan Kütüphaneler

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.api as sm
"# tennis-play-predictor" 
