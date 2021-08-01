import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

twoLevelSubj = "../Data/twoLevelSubj.csv"
df = pd.read_csv(twoLevelSubj)
# print(df.head())

# initial state
init_state = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
print("INITIAL STATE\n",init_state)

# transition matrix for 3 comments  => Second order transition matrix
level1Pairs = df[(df["state_0"].notna()) & (df["state_1"].notna()) & (df["state_2"].notna())]
counts = level1Pairs.groupby(['state_0', 'state_1', 'state_2']).size().unstack()
probs1 = counts.div(counts.sum(axis=1), axis=0)
print("TRANSITION PROBABILITY\n",probs1)

probs1.reset_index(inplace=True) 
probs1 = probs1[probs1.columns[-3:]]

p1 = [init_state] @ probs1
print("NEXT STATE\n",p1)
