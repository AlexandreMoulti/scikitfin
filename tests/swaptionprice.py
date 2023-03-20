import pandas as pd
import numpy as np

df_prices = pd.read_csv("PremiumSwap.csv", sep=";")

df_prices.set_index("Unnamed: 0", inplace=True)

array_of_tuple = []

for maturity in df_prices.index:
    for tenor in df_prices.columns:
        array_of_tuple.append((maturity, int(tenor), df_prices.loc[maturity, tenor]))

array_of_tuple = np.array(array_of_tuple, dtype=[('maturity', "i4"), ("tenor", "i4"), ("value", "f8")])[:5]

print(array_of_tuple)