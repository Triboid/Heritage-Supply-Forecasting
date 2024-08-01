#This file is intended to make plots while making the streamlit app

import matplotlib.pyplot  as plt
import pandas as pd
import numpy as np

df = pd.read_csv("Datasets/Milk Procurement.csv")
df['Date'] = df['Year'] + ' '+ df['Quarter']
df['Date'] = [i[2:] for i in df['Date']]

df.set_index('Date')
data = df['Milk Procurement (in MLPD)']
data1 = data.diff()[1:].diff()[1:]

plt.plot(data[4:],label= 'Milk Procurement (in MLPD)')
plt.xticks(np.arange(5, 22, step=1))
plt.title('Milk Procurement volume from 2020-21 to Q1 2024-25')
plt.legend()
plt.show()