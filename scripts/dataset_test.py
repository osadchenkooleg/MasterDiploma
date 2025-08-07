from datasets import load_from_disk

ds = load_from_disk("../data/raw/bigclonebench/java")
print("Rows:", len(ds), "Пример:", ds[0].keys())
