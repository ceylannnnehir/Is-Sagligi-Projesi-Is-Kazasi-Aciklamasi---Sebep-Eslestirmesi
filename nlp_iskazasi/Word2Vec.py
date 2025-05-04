import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import os
import time

# NLTK verilerini indir (ilk çalıştırmada gerekebilir)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# merge.txt dosyasından veriyi oku
with open("merge.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Cümlelere ayır
sentences = sent_tokenize(text)

# NLP araçlarını başlat
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Ön işleme fonksiyonu
def preprocess_sentence(sentence):
    tokens = word_tokenize(sentence)
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return lemmatized_tokens, stemmed_tokens

# Tüm cümleleri işle
tokenized_corpus_lemmatized = []
tokenized_corpus_stemmed = []

for sentence in sentences:
    lemmatized_tokens, stemmed_tokens = preprocess_sentence(sentence)
    tokenized_corpus_lemmatized.append(lemmatized_tokens)
    tokenized_corpus_stemmed.append(stemmed_tokens)

# Word2Vec parametre kombinasyonları
parameters = [
    {'model_type': 'cbow', 'window': 2, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 2, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 300},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 300}
]

# Eğitim ve kayıt fonksiyonu (tekli)
def train_and_save_model(corpus, params, model_name):
    model = Word2Vec(
        corpus,
        vector_size=params['vector_size'],
        window=params['window'],
        min_count=1,
        sg=1 if params['model_type'] == 'skipgram' else 0
    )
    filename = f"{model_name}_{params['model_type']}_window{params['window']}_dim{params['vector_size']}.model"
    model.save(filename)
    print(f"Model saved: {filename}")
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    return size_mb

    
# 16 modeli eğit ve kaydet

# Lemmatized modellerin toplam süresi ve boyutu
start_time_lemma = time.time()
total_size_lemma = 0
for param in parameters:
    total_size_lemma += train_and_save_model(tokenized_corpus_lemmatized, param, "lemmatized_model")
end_time_lemma = time.time()
elapsed_lemma = end_time_lemma - start_time_lemma

# Stemmed modellerin toplam süresi ve boyutu
start_time_stem = time.time()
total_size_stem = 0
for param in parameters:
    total_size_stem += train_and_save_model(tokenized_corpus_stemmed, param, "stemmed_model")
end_time_stem = time.time()
elapsed_stem = end_time_stem - start_time_stem

# Sonuçları yazdır
print("\nModel Eğitimi Özeti:")
print(f"Lemmatized modeller için toplam süre: {elapsed_lemma:.2f} saniye")
print(f"Lemmatized modellerin toplam boyutu: {total_size_lemma:.2f} MB\n")
print(f"Stemmed modeller için toplam süre: {elapsed_stem:.2f} saniye")
print(f"Stemmed modellerin toplam boyutu: {total_size_stem:.2f} MB")