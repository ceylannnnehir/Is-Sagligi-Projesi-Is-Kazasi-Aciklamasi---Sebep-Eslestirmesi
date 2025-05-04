import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Gerekli NLTK kaynakları (bir kez indirilmesi yeterli)
nltk.download('punkt')

# Ham veri dosyasını oku
with open("merge.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenization (sadece alfabetik kelimeleri al)
tokens = [token.lower() for token in word_tokenize(text) if token.isalpha()]

# Kelime frekanslarını say
word_freq = Counter(tokens)

# Sıralı frekans listesi (en sık olandan az sıka)
sorted_freq = word_freq.most_common()

# Rank ve frekans değerlerini ayıkla
ranks = np.arange(1, len(sorted_freq) + 1)
frequencies = np.array([freq for _, freq in sorted_freq])

# Log-Log grafiği çiz
plt.figure(figsize=(10, 6))
plt.loglog(ranks, frequencies, marker=".")
plt.title("Zipf Yasası - Ham Veri Üzerinden")
plt.xlabel("Kelime Sırası (Rank) [log]")
plt.ylabel("Kelime Frekansı [log]")
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.tight_layout()

# 🔽 Grafik dosyasını kaydet (PNG formatında)
plt.savefig("zipf_plot.png", dpi=300)

# Grafiği göster
plt.show()

# Kelime sayısı bilgisi
print(f"Toplam alfabetik kelime sayısı: {len(tokens)}")
print(f"Benzersiz kelime sayısı: {len(word_freq)}")
print("Grafik 'zipf_plot.png' olarak kaydedildi.")
