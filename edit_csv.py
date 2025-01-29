import pandas as pd

file_path = "tabela_kalorycznosci.csv"
df = pd.read_csv(file_path)

produkt = 'kurczak / kaczka'
new_values =[
    1.1,    # Gęstość       (g/cm^3)
    0,    # Kalorie       (kcal/100g)
    0,    # Tłuszcz       (g/100g)
    0,    # Węglowodany   (g/100g)
    0,    #  w tym cukry  (g/100g)
    0,    # Białko        (g/100g)
    0     # Sól           (g/100g)
]


df.iloc[df.index[df["nazwa"] == produkt],2:] = new_values
print(df.loc[df["nazwa"] == produkt])

df.to_csv("tabela_kalorycznosci.csv", index=False)  # Replace with desired output file name