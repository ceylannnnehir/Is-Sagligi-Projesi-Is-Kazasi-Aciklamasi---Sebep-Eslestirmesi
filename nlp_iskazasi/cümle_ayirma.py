import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import csv

# Gerekli NLTK kaynaklarını indir (İlk çalıştırmada gerekebilir)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# merge.txt dosyasından içerik oku
with open("merge.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Cümlelere ayır
sentences = sent_tokenize(text)

# Stopwords listesi
stop_words = set(stopwords.words('english'))

# Lemmatizer ve Stemmer başlat
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# İlk 5 cümleyi işle
print("\n\n===== İLK 5 CÜMLE İŞLENİYOR =====\n")
for i in range(min(5, len(sentences))):
    print(f"\n--- Cümle {i+1} ---")
    print("Orijinal:", sentences[i])
    
    # Tokenization
    tokens = word_tokenize(sentences[i])
    print("Tokenization:", tokens)
    
    # Lowercasing + Alphabetic filtre
    lower_tokens = [token.lower() for token in tokens if token.isalpha()]
    print("Lowercasing:", lower_tokens)
    
    # Stopword removal
    filtered_tokens = [token for token in lower_tokens if token not in stop_words]
    print("Stopword Removal:", filtered_tokens)
    
    # Lemmatization
    lemmatized = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    print("Lemmatization:", lemmatized)
    
    # Stemming
    stemmed = [stemmer.stem(token) for token in filtered_tokens]
    print("Stemming:", stemmed)

# Ön işleme fonksiyonu
def preprocess_sentence(sentence):
    tokens = word_tokenize(sentence)
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return lemmatized_tokens, stemmed_tokens

# Başlangıçtaki toplam kelime sayısını hesapla (tüm metin için, sadece alfabetik kelimeler)
all_tokens = word_tokenize(text)
initial_word_count = len([token for token in all_tokens if token.isalpha()])

# Cümleleri işle + işlenmiş kelime sayısını topla
processed_word_count = 0

# Cümleleri işle
tokenized_corpus_lemmatized = []
tokenized_corpus_stemmed = []

for sentence in sentences:
    lemmatized_tokens, stemmed_tokens = preprocess_sentence(sentence)
    tokenized_corpus_lemmatized.append(lemmatized_tokens)
    tokenized_corpus_stemmed.append(stemmed_tokens)
    processed_word_count += len(lemmatized_tokens)  # ikisi aynı uzunlukta olduğu için biri yeterli

# Lemmatized sonuçları CSV'ye kaydet
with open("lemmatized_sentences.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    for tokens in tokenized_corpus_lemmatized:
        writer.writerow([' '.join(tokens)])

# Stemmed sonuçları CSV'ye kaydet
with open("stemmed_sentences.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    for tokens in tokenized_corpus_stemmed:
        writer.writerow([' '.join(tokens)])



import matplotlib.pyplot as plt
from collections import Counter
import math

# Zipf grafiği çizen ve dosya kaydeden fonksiyon
def draw_zipf_plot(words, title, filename):
    word_freq = Counter(words)
    sorted_freq = sorted(word_freq.values(), reverse=True)

    plt.figure(figsize=(10, 6))
    plt.plot([math.log(rank + 1) for rank in range(len(sorted_freq))],
             [math.log(freq) for freq in sorted_freq])
    plt.xlabel("Kelime Sırası (Rank) [log]")
    plt.ylabel("Kelime Frekansı [log]")
    plt.title(f"Zipf Yasası Grafiği - {title}")
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    return len(word_freq), sum(word_freq.values())

# CSV'den kelimeleri oku
def load_words_from_csv(file_path):
    words = []
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            words.extend(row[0].split())
    return words

# Zipf grafikleri oluştur ve kaydet
lemmatized_words = load_words_from_csv("lemmatized_sentences.csv")
lemma_unique, lemma_total = draw_zipf_plot(lemmatized_words, "Lemmatized Veri", "zipf_lemmatized.png")

stemmed_words = load_words_from_csv("stemmed_sentences.csv")
stem_unique, stem_total = draw_zipf_plot(stemmed_words, "Stemmed Veri", "zipf_stemmed.png")

# Özet Bilgi
print("\n===== VERİ BOYUTU BİLGİSİ =====")
print(f"Başlangıçtaki (ham) toplam alfabetik kelime sayısı : {initial_word_count}")
print(f"Stopword sonrası (işlenmiş) kelime sayısı           : {processed_word_count}")
print(f"Lemmatized veri toplam kelime sayısı               : {lemma_total}")
print(f"Lemmatized veri eşsiz kelime sayısı                : {lemma_unique}")
print(f"Stemmed veri toplam kelime sayısı                  : {stem_total}")
print(f"Stemmed veri eşsiz kelime sayısı                   : {stem_unique}")

kelime_azalmasi_lemma = initial_word_count - lemma_total
kelime_azalmasi_stem = initial_word_count - stem_total

print(f"\nHam veriden lemmatization sonrası çıkarılan kelime sayısı: {kelime_azalmasi_lemma}")
print(f"Ham veriden stemming sonrası çıkarılan kelime sayısı     : {kelime_azalmasi_stem}")