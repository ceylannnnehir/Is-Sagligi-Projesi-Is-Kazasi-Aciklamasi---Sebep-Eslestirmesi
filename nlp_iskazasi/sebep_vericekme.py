from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup, NavigableString
import time, os

# Kayıt klasörü
SAVE_DIR = "accident_reports"
os.makedirs(SAVE_DIR, exist_ok=True)

# ID listesi (istediğin kadar ekleyebilirsin)
ids = [
    "169298.015", "169112.015", "168946.015", "168908.015", "168911.015",
    "169200.015", "169526.015", "169580.015", "169690.015", "169679.015",
    "169574.015","168805.015","169222.015","168840.015","171749.015",
    "168742.015","170271.015", "168637.015"," 172034.015", "168336.015",
    "168340.015", "168298.015","168721.015", "168257.015","168113.015",
    "168179.015", "168523.015","168268.015", "168250.015","168105.015",
    "168546.015","168256.015","170763.015","168077.015","168551.015",
    "167983.015","168030.015","167984.015","167694.015","167523.015",
    "162609.015","164782.015"
]

# Tarayıcıyı başlat
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

#  Her ID için işlem
for accident_id in ids:
    url = f"https://www.osha.gov/ords/imis/accidentsearch.accident_detail?id={accident_id}"
    driver.get(url)
    time.sleep(3)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    #  Summary
    summary = ""
    summary_tag = soup.find("p", style="font-size: 110%")
    if summary_tag:
        summary = summary_tag.get_text(strip=True)

    #  Abstract
    abstract = ""
    for div in soup.find_all("div"):
        strong_tag = div.find("strong")
        if strong_tag and "Abstract:" in strong_tag.text:
            content = ""
            for sibling in strong_tag.next_siblings:
                if isinstance(sibling, NavigableString):
                    content += sibling.strip()
            abstract = content.strip()
            break

    #  Keywords
    keywords = ""
    for div in soup.find_all("div"):
        strong_tag = div.find("strong")
        if strong_tag and "Keywords:" in strong_tag.text:
            content = ""
            for sibling in strong_tag.next_siblings:
                if isinstance(sibling, NavigableString):
                    content += sibling.strip()
            keywords = content.strip()
            break

    #  Sonuçları birleştir
    result = ""
    if summary:
        result += f"{summary}\n\n"
    if abstract:
        result += f"Abstract: {abstract}\n\n"
    if keywords:
        result += f"Keywords: {keywords}\n"

    #  Dosyaya yaz
    if result.strip():
        filename = f"{accident_id.replace('.', '_')}.txt"
        with open(os.path.join(SAVE_DIR, filename), "w", encoding="utf-8") as f:
            f.write(result)
        print(f" Kaydedildi: {filename}")
    else:
        print(f"Veri bulunamadı: {accident_id}")

# Tarayıcıyı kapat
driver.quit()
print(" Tüm veriler başarıyla çekildi ve kaydedildi.")
