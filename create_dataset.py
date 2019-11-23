from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
x, y = make_classification(n_samples=20000, 
                           n_features=200, 
                           n_informative=100)

xt, xv, yt, yv = train_test_split(x, y, test_size=0.6)

with open("data/generated.train.txt", "w") as f:
    for i in range(xt.shape[0]):
        x_line = xt[i, :].tolist()
        x_line = ["{0}:{1:.6f}".format(j+1, value) for j, value in enumerate(x_line)]
        line = str(yt[i]) + " " + " ".join(x_line) + "\n"
        f.write(line)
    
with open("data/generated.test.txt", "w") as f:
    for i in range(xv.shape[0]):
        x_line = xv[i, :].tolist()
        x_line = ["{0}:{1:.6f}".format(j+1, value) for j, value in enumerate(x_line)]
        line = str(yv[i]) + " " + " ".join(x_line) + "\n"
        f.write(line)
