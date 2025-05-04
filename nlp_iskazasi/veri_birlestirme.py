import os

#  Kayıt klasörü ve çıktı dosyası
SAVE_DIR = "accident_reports"
OUTPUT_FILE = "merge.txt"
INHIBITOR_FILE = "inhibitor.txt"

# inhibitor.txt içeriğini oku
with open(INHIBITOR_FILE, "r", encoding="utf-8") as f:
    inhibitor_content = f.read().strip()

# .txt dosyalarını sırayla al
txt_files = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith(".txt")])

# Hepsini birleştir
with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    for i, filename in enumerate(txt_files):
        filepath = os.path.join(SAVE_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as infile:
            accident_id = filename.replace("_", ".").replace(".txt", "")
            outfile.write(f"--- ID: {accident_id} ---\n")
            outfile.write(infile.read().strip())
            outfile.write("\n\n")
        
        # En son ID'den sonra OSHA içeriğini ve inhibitor.txt'yi yaz
        if i == len(txt_files) - 1:
            outfile.write(">>> OSHA PREVENTIVE ANALYSIS TEXTS <<<\n")
            outfile.write(inhibitor_content)
            outfile.write("\n")

print(f" merge.txt dosyası oluşturuldu. OSHA içeriği yalnızca en sona eklendi.")
