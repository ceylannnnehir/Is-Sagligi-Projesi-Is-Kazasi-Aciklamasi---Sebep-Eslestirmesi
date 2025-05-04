from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# merge.txt dosyasından veriyi oku
with open("merge.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Cümlelere ayırma
sentences = sent_tokenize(text)

# Lemmatizer ve Stemmer'ı başlat
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Stopwords listesini almak
stop_words = set(stopwords.words('english'))

# Ön işleme fonksiyonu
def preprocess_sentence(sentence):
    tokens = word_tokenize(sentence)
    filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return lemmatized_tokens, stemmed_tokens

# Her cümleyi işleyelim
tokenized_corpus_lemmatized = []
tokenized_corpus_stemmed = []

for sentence in sentences:
    lemmatized_tokens, stemmed_tokens = preprocess_sentence(sentence)
    tokenized_corpus_lemmatized.append(lemmatized_tokens)
    tokenized_corpus_stemmed.append(stemmed_tokens)

##########################################
# Lemmatize edilmiş cümle listesi (önceden hazırlanmış olduğunu varsayıyoruz)
lemmatized_texts = [' '.join(tokens) for tokens in tokenized_corpus_lemmatized]

# TF-IDF vektörleştirme
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(lemmatized_texts)
feature_names = vectorizer.get_feature_names_out()

# Pandas DataFrame'e çevirme (isteğe bağlı, analiz kolaylığı sağlar)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
print(tfidf_df.head())

# Güvenlik önerisi sözlüğü
recommendations = {
    'fall': 'Öneri: Fall Protection Training (OSHA 1926.503)',
    'ladder': 'Öneri: Ladder Safety Standards (OSHA 1926.1053)',
    'protective': 'Öneri: PPE Compliance and Hazard Assessment (OSHA 1926.102)',
    'clothing': 'Öneri: PPE Compliance and Hazard Assessment (OSHA 1926.102)',
}

# Anahtar kelimeler
target_words = ['fall', 'ladder', 'protective', 'clothing']

# Cosine similarity hesapla ve önerileri yazdır
for target_word in target_words:
    if target_word in feature_names:
        target_index = feature_names.tolist().index(target_word)
        target_vector = tfidf_matrix[:, target_index].toarray()
        similarities = cosine_similarity(target_vector.T, tfidf_matrix.toarray().T).flatten()
        top_5_indices = similarities.argsort()[-6:][::-1]  # kendisi dahil

        print(f"\n '{target_word}' kelimesine en benzer 5 kelime (TF-IDF cosine similarity):")
        for index in top_5_indices:
            print(f"- {feature_names[index]}: {similarities[index]:.4f}")
        
        # Güvenlik önerisini yazdır
        if target_word in recommendations:
            print(f"{recommendations[target_word]}")
    else:
        print(f"'{target_word}' kelimesi TF-IDF matrisinde bulunamadı.")
