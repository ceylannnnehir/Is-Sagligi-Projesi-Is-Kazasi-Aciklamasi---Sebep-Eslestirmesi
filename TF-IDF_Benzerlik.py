from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import pandas as pd

# CSV'den cümleleri oku
def load_sentences(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        return [row[0] for row in csv.reader(f)]

# En benzer 5 sonucu bul ve puanla
def top_5_similar_sentences_tfidf(sentences, query_index):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    query_vector = tfidf_matrix[query_index]
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][1:6]  # kendisini çıkar
    results = []
    for i in top_indices:
        score = similarities[i]
        if score >= 0.90:
            puan = 5
        elif score >= 0.75:
            puan = 4
        elif score >= 0.60:
            puan = 3
        elif score >= 0.45:
            puan = 2
        else:
            puan = 1
        results.append((f"doc{i}", sentences[i], score, puan))
    return results

def tfidf_similarity_pipeline(csv_path, query_index, name):
    sentences = load_sentences(csv_path)
    top5 = top_5_similar_sentences_tfidf(sentences, query_index)

    print(f"\n{name.upper()} için Giriş Cümlesi:\n[{query_index}] {sentences[query_index]}\n")

    #print("Benzerlik Aralığı Açıklaması:")
    #print("  0.95 - 1.00 : 5 → Çok güçlü benzerlik, aynı tema")
    #print("  0.75 - 0.95 : 4 → Anlamlı ve açık benzerlik")
    #print("  0.60 - 0.75 : 3 → Ortalama düzeyde benzer")
    #print("  0.45 - 0.60 : 2 → Kısmen ilgili ama bağlam zayıf")
    #print("  0.00 - 0.45 : 1 → Çok alakasız\n")

    table = []
    total_score = 0
    for idx, sent, sim, score in top5:
        print(f"[{idx}] ({sim:.4f}) ")
        print(f"   {sent}")
        table.append({
            "Cümle No": idx,
            "Metin": sent,
            "Benzerlik Skoru": round(sim, 4),            
            "Puan": score
            
        })
        total_score += score

    avg_score = round(total_score / len(top5), 2)

    return table, avg_score


# Giriş cümlesi indexini seç
query_index = 7

# Her iki dosya için çalıştır
lemm_results, lemm_avg = tfidf_similarity_pipeline("lemmatized_sentences.csv", query_index, "Lemmatized")
stem_results, stem_avg = tfidf_similarity_pipeline("stemmed_sentences.csv", query_index, "Stemmed")

# Pandas ile sonuçları tabloya dök
df_lemm = pd.DataFrame(lemm_results)
df_stem = pd.DataFrame(stem_results)

print("\n\n--- TF-IDF SONUÇ TABLOSU ---\n")
print(">> Lemmatized TF-IDF Sonuçları:\n")
print(df_lemm.to_markdown(index=False))
print(f"Ortalama Puan: {lemm_avg}")

print("\n>> Stemmed TF-IDF Sonuçları:\n")
print(df_stem.to_markdown(index=False))
print(f"Ortalama Puan: {stem_avg}")
