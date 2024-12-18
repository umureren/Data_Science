
# UMUR EREN ÖZDEMİR 201220012


import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def preprocess_data(data):
    # Eksik verileri sütun mod değeri ile doldur
    data = data.drop('Cihaz Ağırlığı', axis=1)
    for col in data.columns:
        mode_value = data[col].mode().iloc[0]
        data[col].fillna(mode_value, inplace=True)
    
    # Veri çerçevesini masaüstüne CSV dosyası olarak kaydetme
    data.to_excel(r'C:\Users\user\Desktop\preprocessed_data.xlsx', index=False)

    # Tüm sütunlara label encoding uygulama
    for col in data.columns:
        if data[col].dtype == 'object':  # Sadece kategorik sütunlara işlem uygula
            data[col] = LabelEncoder().fit_transform(data[col].astype(str))
    print(data)
    return data

def preprocess_and_visualize_data(data):
    
    temp1_data = data.copy()
    
    record_count = len(temp1_data)
    print("\nKayıt Sayısı:", record_count)

    # Veri setinin nitelik sayısı
    nitelik_sayisi = len(temp1_data.columns)
    print(f"Veri setinin nitelik sayısı: {nitelik_sayisi}")

    # Veri setinin nitelik tipleri
    nitelik_tipleri = temp1_data.dtypes
    print("Veri setinin nitelik tipleri:")
    print(nitelik_tipleri)



    # Ram sütununun dağılımını gösterme
    temp1_data['Ram (Sistem Belleği)'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('Ram (Sistem Belleği)')
    plt.show()



    sutunlar = ['Ram (Sistem Belleği)', 'Bellek Hızı']
    for sutun in sutunlar:
        temp1_data[sutun] = temp1_data[sutun].replace(r'[^\d.,]', '', regex=True)
        temp1_data[sutun] = pd.to_numeric(temp1_data[sutun].str.replace(',', ''), errors='coerce')


    # Korelasyon grafiği
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Bellek Hızı', y='Ram (Sistem Belleği)', data=temp1_data)
    plt.title('Bellek Hızı ve Ram (Sistem Belleği) Korelasyon Grafiği')
    plt.show()

    # İlgili sütunlara yapılan işlemleri genişletme
    columns_to_expand = ['Ram (Sistem Belleği)','Bellek Hızı']
    for col in columns_to_expand:
        # Orijinal veri üzerinde işlemleri gerçekleştirme
        print(f"\nStatistics for column '{col}':")
        print(f"Number of records: {len(temp1_data)}")
        print(f"Number of unique values: {temp1_data[col].nunique()}")
       

        if pd.api.types.is_numeric_dtype(temp1_data[col]):
            print(f"Mean: {temp1_data[col].mean()}")
            print(f"Median: {temp1_data[col].median()}")
            print(f"Standard Deviation: {temp1_data[col].std()}")
            print(f"Variance: {temp1_data[col].var()}") 
            print(f"Minimum: {temp1_data[col].min()}")
            print(f"25th Percentile (Q1): {temp1_data[col].quantile(0.25)}")
            print(f"50th Percentile (Q2 or Median): {temp1_data[col].quantile(0.50)}")
            print(f"75th Percentile (Q3): {temp1_data[col].quantile(0.75)}")
            print(f"Maximum: {temp1_data[col].max()}")
        else:
            print("The column does not contain numeric values.")

    return temp1_data

def classify_data(features, target):
   
    for _ in range(1):  
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=None)  # random_state=None her seferinde farklı parçalama yapar

        # Gradient Boosting sınıflandırma modelini oluşturma
        gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.15, max_depth=3, random_state=42)
        gb_model.fit(X_train, y_train)

        # Test verisi üzerinde tahmin yapma
        y_pred = gb_model.predict(X_test)

        # Konfüzyon matrisini oluşturma
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)

        # Doğruluk yüzdesini hesapla
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        with open("trained_model.pkl", "wb") as file:
         pickle.dump(gb_model, file)


# data.xlsx dosyasını okuma
data = pd.read_excel("C:/Users/user/Desktop/data.xlsx")


temp_data = data.copy()
preprocess_and_visualize_data(temp_data)
# Verileri işle ve sınıflandır
preprocessed_data = preprocess_data(temp_data)
classify_data(preprocessed_data.drop('Fiyat', axis=1), preprocessed_data['Fiyat'])


