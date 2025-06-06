# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Fzs3g-0Qvflie4vxHqfDCmSekHR0oT4t
"""

import pandas as pd
data = {
    'Patient_ID': [1, 2, 3, 4, 5],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Cholesterol': [180, 220, 190, 250, 210]
}
df = pd.DataFrame(data)
threshold = 200
high_cholesterol_patients = df[df['Cholesterol'] > threshold]
print(high_cholesterol_patients)