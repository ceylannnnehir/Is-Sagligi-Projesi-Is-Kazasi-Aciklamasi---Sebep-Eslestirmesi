import requests
from bs4 import BeautifulSoup

url = "https://workplacesafety.com/top-10-osha-violations-2022/"

# Sayfayı indir
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Hedef başlıklar (numaralarıyla eşle)
target_titles = {
    "9. Personal Protective Equipment: 1926.102",
    "8. Fall Protection (Training): 1926.503",
    "4. Ladders: 1926.1053"
}

# Sonuç listesi
results = []

# Sayfada paragrafları tararken başlığı ve altındaki içeriği eşleştir
elements = soup.find_all(["h2", "h3", "p", "li"])
i = 0
while i < len(elements):
    text = elements[i].get_text(strip=True)
    if text in target_titles:
        # Başlık bulundu
        content = text + "\n"
        # Sonraki öğe açıklama olabilir
        j = i + 1
        while j < len(elements):
            next_text = elements[j].get_text(strip=True)
            # Eğer yeni bir başlık gibi görünüyorsa, dur
            if any(t in next_text for t in target_titles) or next_text.startswith(tuple(str(n) + '.' for n in range(1, 11))):
                break
            content += next_text + "\n"
            j += 1
        results.append(content.strip())
        i = j
    else:
        i += 1

# 📄 Dosyaya yaz
with open("inhibitor.txt", "w", encoding="utf-8") as f:
    for entry in results:
        f.write(entry + "\n\n")

print("✅ Başlıklar ve açıklamaları 'inhibitor.txt' dosyasına kaydedildi.")
