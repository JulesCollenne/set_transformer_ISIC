import pandas as pd

if __name__ == "__main__":
    model = "CNN"
    for leset in ("train", "val", "test"):
        df = pd.read_csv(f"features/{model}_{leset}.csv")
        df['target'] = df['label'].apply(lambda x: (x + 1) % 2)
        # df['target'] = df['image_name'].apply(lambda x: 1 if x.startswith('mel/') else 0)
        df['image_name'] = df['image_name'].str.replace('mel/', '').str.replace('nev/', '')
        df['image_name'] = df['image_name'].str.replace('.jpg', '')
        df.to_csv(f"features/{model}_{leset}.csv", index=False)
