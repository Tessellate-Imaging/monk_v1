import os
import sys
import numpy as np 
import pandas as pd 

"""
headers = ["ID", "LABEL"];
id_label = [];
for i in range(100):
	img = str(i) + ".jpg";
	if(i<=49):
		label = "dog";
	else:
		label = "cat";
	id_label.append([img, label])

df = pd.DataFrame(id_label, columns=headers)

df.to_csv("train.csv", index=False)
"""

"""
headers = ["ID", "LABEL"];
id_label = [];
for i in range(50):
	img = str(i) + ".jpg";
	if(i<=24):
		label = "cat";
	else:
		label = "dog";
	id_label.append([img, label])

df = pd.DataFrame(id_label, columns=headers)

df.to_csv("val.csv", index=False)
"""


headers = ["ID", "LABEL"];
id_label = [];
for i in range(28):
	img = str(i) + ".jpg";
	if(i<=13):
		label = "cat";
	else:
		label = "dog";
	id_label.append([img, label])

df = pd.DataFrame(id_label, columns=headers)

df.to_csv("test.csv", index=False)