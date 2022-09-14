from cProfile import label
import pickle 
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 8))

with open("./logs/testmethod4/weight1/iql_iter.pkl", "rb") as f:
    x = pickle.load(f)


l = [0, 2, 3,4, 7, 8, 9, 10]

for i in l:
    with open("./logs/testmethod4/weight{}/iql_val.pkl".format(i), "rb") as f:
        y = pickle.load(f)
        if i < 10:
            plt.plot(x, y, label="weight=0.{}".format(i))
        else:
            plt.plot(x, y, label="weight=1.0")
with open("./logs/iql/Lr01decay/iql_iter.pkl", "rb") as f1:
    with open("./logs/iql/Lr01decay/iql_val.pkl", "rb") as f2:
        x = pickle.load(f1)
        y = pickle.load(f2)
        plt.plot(x, y, label='iql')

plt.legend()
plt.savefig("compare")
plt.show()
