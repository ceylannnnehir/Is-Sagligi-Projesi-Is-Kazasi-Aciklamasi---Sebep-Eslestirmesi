from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# TF-IDF vektörleştirme fonksiyonu
def tfidf_to_csv(tokenized_corpus, filename):
    texts = [' '.join(tokens) for tokens in tokenized_corpus]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    tfidf_df.to_csv(filename, index=False)
    print(f"{filename} kaydedildi.")
    # Pandas DataFrame'e çevirme (isteğe bağlı, analiz kolaylığı sağlar)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    print(tfidf_df.head())
    return tfidf_matrix, feature_names


# Lemma TF-IDF → CSV
tfidf_matrix_lemma, feature_names_lemma = tfidf_to_csv(tokenized_corpus_lemmatized, "tfidf_lemmatized.csv")

# Stem TF-IDF → CSV
tfidf_matrix_stem, feature_names_stem = tfidf_to_csv(tokenized_corpus_stemmed, "tfidf_stemmed.csv")


target_words = ['fall', 'ladder', 'protective','clothing'] 
######### protective clothing bu birlikte yok ayrı ayrı var

print("\n--- Lemma Similarity (Lemmatized) ---")
for target_word in target_words:
    if target_word in feature_names_lemma:
        target_index = np.where(feature_names_lemma == target_word)[0][0]
        target_vector = tfidf_matrix_lemma[:, target_index].toarray()
        similarities = cosine_similarity(target_vector.T, tfidf_matrix_lemma.toarray().T).flatten()
        top_5_indices = similarities.argsort()[-6:][::-1]

        print(f"\n'{target_word}' kelimesine en benzer 5 kelime (Lemmatized TF-IDF cosine similarity):")
        for index in top_5_indices:
            print(f"{feature_names_lemma[index]}: {similarities[index]:.4f}")
    else:
        print(f"'{target_word}' kelimesi TF-IDF matrisinde bulunamadı.")

print("\n--- Cosine Similarity (Stemmed) ---")
for target_word in target_words:
    if target_word in feature_names_stem:
        target_index = np.where(feature_names_stem == target_word)[0][0]
        target_vector = tfidf_matrix_stem[:, target_index].toarray()
        similarities = cosine_similarity(target_vector.T, tfidf_matrix_stem.toarray().T).flatten()
        top_5_indices = similarities.argsort()[-6:][::-1]

        print(f"\n'{target_word}' kelimesine en benzer 5 kelime (Stemmed TF-IDF cosine similarity):")
        for index in top_5_indices:
            print(f"{feature_names_stem[index]}: {similarities[index]:.4f}")
    else:
        print(f"'{target_word}' kelimesi TF-IDF matrisinde bulunamadı.")
