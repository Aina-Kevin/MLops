import pandas as pd

file_path = "data/raw/synthetic_data.xlsx"

df = pd.read_excel(file_path)

new_row = pd.DataFrame({
    "x": [100],
    "y": [260]
})

df = pd.concat([df, new_row], ignore_index=True)

df.to_excel(file_path, index=False)

print("Nouvelle donnée ajoutée.")
print(df.tail())