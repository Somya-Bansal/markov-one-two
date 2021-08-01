import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

twoLevelSubj = "../Data/twoLevelSubj.csv"
df = pd.read_csv(twoLevelSubj)
# print(df.head())

# initial state
init_state1 = np.array([1, 0, 0])
init_state2 = np.array([0, 1, 0])
init_state3 = np.array([0, 0, 1])

# transition matrix for 2 comments
level1Pairs = df[(df["state_0"].notna()) & (df["state_1"].notna())]
print(level1Pairs)
counts = level1Pairs.groupby(['state_0', 'state_1']).size().unstack()
print(counts)
probs1 = (counts / counts.sum())
probs1 = probs1.T
print(probs1)

p1 = [init_state1]
p2 = [init_state2]
p3 = [init_state3]
for i in range(10):
    p1.append(p1[-1] @ probs1)
    p2.append(p2[-1] @ probs1)
    p3.append(p3[-1] @ probs1)

result1 = pd.DataFrame(p1)
result2 = pd.DataFrame(p2)
result3 = pd.DataFrame(p3)
# state_distributions.plot()


fig = plt.figure(figsize=(11,8))

ax1 = fig.add_subplot(311)
result1.plot(ax=ax1, marker='o')
ax2 = fig.add_subplot(312)
result2.plot(ax=ax2, marker='o')
ax3 = fig.add_subplot(313)
result3.plot(ax=ax3, marker='o')

plt.show()
