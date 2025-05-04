# Is-Sagligi-Projesi-Is-Kazasi-Aciklamasi---Sebep-Eslestirmesi

  Bu proje, iş kazalarına dair açıklama metinlerini doğal dil işleme (NLP) ve Word2Vec modelleri ile analiz ederek, kaza nedenlerinin otomatik olarak anlaşılmasını ve sınıflandırılmasını amaçlar. 16 farklı model ile kelime benzerlikleri incelenerek en iyi sonuç veren model tespit edilir.

# Gerekli Kütüphaneler
    matplotlib
    nltk
    csv
    requests
    BeautifulSoup
    selenium.webdriver
    selenium
    ChromeDriverManager
    NavigableString
    pandas
    Scikit-learn
    numpy
    os
    time
    gensim


# Kurulum
    pip install matplotlib
    pip install gensim
    pip install nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    pip install requests
    pip install beautifulsoup4
    pip install selenium
    pip install chromedriver-autoinstaller
    pip install pandas
    pip install scikit-learn
    pip install numpy


# İlk kez çalıştırıyorsanız, aşağıdaki NLTK modüllerini de indirmeniz gerekir:
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Veri Seti
  Kaynak: OSHA (Occupational Safety and Health Administration)
  Bu veri seti, Amerika Birleşik Devletleri İş Güvenliği ve Sağlığı İdaresi (OSHA) tarafından yayımlanan iş kazası raporlarına dayanmaktadır. Kazaların açıklama metinleri içerir ve metin madenciliği/NLP uygulamaları için uygundur.
  Projede kullanılan merge.txt dosyası, bu açıklamalardan oluşturulmuş metinlerin birleşiminden oluşur.
  
  Amaç: İş kazası açıklama metinlerinde geçen kelimelerin semantik yakınlıklarını bulmak ve kazaların nedenlerini belirlemeye yardımcı olacak kelime ilişkilerini modellemek ve metinlerden otomatik çıkarım yapılmasını sağlamak.

# Kullanılan Yöntemler
  1. Word2Vec Modelleri
  Bu projede CBOW ve Skip-gram mimarileri ile 16 farklı Word2Vec modeli eğitilmiştir. Her model; pencere boyutu (window), vektör boyutu (vector_size) ve ön işleme yöntemi (lemmatizasyon/stemming) gibi farklı parametre kombinasyonları ile oluşturulmuştur. Modellerin eğitimi merge.txt adlı veri dosyasından elde edilen cümleler üzerinden yapılmıştır.
  
  2. TF-IDF (Term Frequency - Inverse Document Frequency)
  TF-IDF yaklaşımı, kelimelerin belgelere özgüllüğünü hesaplayarak önemli anahtar kelimeleri ortaya çıkarmada kullanılmıştır. Bu yöntem, her bir kelimenin belgelerdeki frekansına ve diğer belgelerdeki yaygınlığına göre ağırlık verir.
  
  Bu yöntem sayesinde sık kullanılan ama bilgi değeri düşük kelimeler elenirken, nadir geçen ama önemli olan kelimeler öne çıkarılır.

  TF-IDF çıktıları, Word2Vec ile elde edilen benzerlik skorlarıyla karşılaştırılmıştır.

# Model Nasıl Oluşturulur? (Adım Adım)
1- Veri Hazırlığı:
  merge.txt dosyası okunur ve cümlelere ayrılır. Cümleler hem lemmatize hem de stem edilerek iki ayrı veri kümesi oluşturulur.

2- Önişleme:

  Cümleler tokenize edilir.
  
  Stopword’ler temizlenir.
  
  Lemmatization ve stemming işlemleri yapılır.

3- Model Parametreleri:
  Toplam 16 farklı model eğitilir:

    2 model türü: CBOW ve Skip-gram
    
    2 pencere boyutu: 2 ve 4
    
    2 vektör boyutu: 100 ve 300
    
    2 ön işleme: Lemmatized ve Stemmed

4- Model Eğitimi:
  Her parametre kombinasyonu için bir Word2Vec modeli eğitilir ve .model uzantısıyla diske kaydedilir.

5- Örnek Sorgulama:
  Eğitilen modeller üzerinde belirli anahtar kelimeler (örneğin: accident, fall) ile most_similar() fonksiyonu çağrılarak en benzer 5 kelime elde edilir.

# Kullanım Senaryosu
  İş kazalarının açıklama metinlerinden kaza nedenlerini otomatik olarak tahmin etmek

  Kaza türlerine göre önlem stratejileri geliştirmek

  İş güvenliği raporları üretmek ve metin madenciliği yapmak

