import os

def read_reports_from_folder(folder_path, limit=None):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")
    
    mapp = {}
    for i, filename in enumerate(os.listdir(folder_path)):
        if limit is not None and i >= limit:
            break
        if filename.endswith(".txt"):
            try:
                with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                    report_text = f.read()
                    mapp[filename] = report_text
            except Exception as e:
                print(f"Could not read {filename}: {e}")
                continue
    return mapp


mapp = read_reports_from_folder("data/Train", limit=2)
print(mapp.keys())