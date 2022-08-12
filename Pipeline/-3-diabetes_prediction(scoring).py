# End-to-End Diabetes Machine Learning Pipeline III

# Calisma bitti ve bunu biriyle paylastik; yeni veriler veri seti icerisine eklenmis

import joblib
import pandas as pd

df = pd.read_csv("datasets/diabetes.csv")

# Rastgele hasta secelim;
random_user = df.sample(1, random_state=45)

# Modeli cagiralim;
new_model = joblib.load("voting_clf.pkl")

# Modele gozlem birimini sorduk.
new_model.predict(random_user)                      # Boyutlarin uyusmadigina dair uyari aldik.


# Veri serindeki feature'larda degisiklik oldu. Yeni hastanin bilgileri artik eski yapimiza uyumlu degil.
# Veri setinin eski veri setiyle uyumlu olmasi; donusturulmesi lazim.

from diabetes_pipeline import diabetes_data_prep

X, y = diabetes_data_prep(df)

# Rastgele hasta secelim;
random_user = X.sample(1, random_state=50)

# Modeli cagiralim;
new_model = joblib.load("voting_clf.pkl")

# Yeni modele gozlem birimini tekrar sorduk.
new_model.predict(random_user)                      # Tahmin sonucunu elde etmis olduk. :)