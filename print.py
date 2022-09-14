import matplotlib.pyplot as plt
import pickle
with open("logs/testmethod4/weight2/iql_iter.pkl", "rb") as f:
    it = pickle.load(f)
with open("logs/testmethod4/weight2/iql_val.pkl", "rb") as f:
    val = pickle.load(f)
plt.plot(it, val)
plt.ylim(15, 20)
plt.show()