# 🐦 Kuş Sınıflandırma

Bu proje, kuş resimlerini sınıflandırmak için bir makine öğrenimi modeli kullanır. Veri seti, farklı kuş türlerinin resimlerini içeren bir klasör yapısında bulunmalıdır. Proje, bu resimleri kullanarak bir `KNeighborsClassifier` modeli eğitir ve sınıflandırma performansını değerlendirir.

## 📁 Veri Seti Yapısı

Veri seti, "Birds_Classification" adında bir klasör içinde tutulmalıdır. Bu klasör altında, her kuş türü için ayrı bir alt klasör olmalıdır. Örneğin:

Birds_Classification/  
│  
├── 🐦 KuşTürü1/  
│   ├── 📷 resim1.jpg  
│   ├── 📷 resim2.jpg  
│   └── ...  
│  
├── 🐤 KuşTürü2/  
│   ├── 📷 resim1.jpg  
│   ├── 📷 resim2.jpg  
│   └── ...  
│  
└── ...

## 📋 Gereksinimler

Bu projeyi çalıştırmak için şu yazılım ve kütüphanelere ihtiyacınız vardır:

- Python 3.x
- scikit-learn
- NumPy
- scikit-image

Gerekli kütüphaneleri yüklemek için terminal veya komut istemcisinde şu komutları kullanabilirsiniz:

```bash
pip install scikit-learn numpy scikit-image

▶️ Kullanım
Proje dosyalarını indirin veya kopyalayın.
Veri setini uygun bir şekilde hazırlayın ve "Birds_Classification" adlı bir klasör içine yerleştirin.
bird_classification.py dosyasını çalıştırarak modeli eğitin ve değerlendirin.

Örnek Kullanım:
python bird_classification.py

📊 Değerlendirme Metrikleri
Model performansını değerlendirmek için aşağıdaki metrikler kullanılır:

Doğruluk (Accuracy): Doğru sınıflandırılan örneklerin oranı.
Hassasiyet (Precision): Pozitif olarak tahmin edilen örneklerin gerçekten pozitif olan örneklerin oranı.
Duyarlılık (Recall): Gerçekten pozitif olan örneklerin tespit edilme oranı.
F1 Skoru: Hassasiyet ve duyarlılık arasındaki dengeyi gösteren bir metrik.

Output:
Model Accuracy: 0.875
Precision: 0.8977272727272727
Recall: 0.875
F1 Score: 0.8708333333333333
