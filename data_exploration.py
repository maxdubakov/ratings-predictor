import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)

data_android = pd.read_csv('googleplaystore.csv', sep=',')

# print(data_android.groupby('Android Ver').count()['Type'])
# print(data_android.columns)
#
# print(data_android.head())
# print(data_android.groupby('Type').count()['Rating'])
# print('-' * 40)

print(data_android[data_android['Rating'] < 3].count())

plt.show()
