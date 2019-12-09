from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
 
def generate(name, **k):
    x, y = make_classification(**k)
    xt, xv, yt, yv = train_test_split(x, y, test_size=0.1)
    with open("data/{}.train.txt".format(name), "w") as f:
        for i in range(xt.shape[0]):
            x_line = xt[i, :].tolist()
            x_line = ["{0}:{1:.6f}".format(j+1, value) for j, value in enumerate(x_line)]
            line = str(yt[i]) + " " + " ".join(x_line) + "\n"
            f.write(line)
        
    with open("data/{}.test.txt".format(name), "w") as f:
        for i in range(xv.shape[0]):
            x_line = xv[i, :].tolist()
            x_line = ["{0}:{1:.6f}".format(j+1, value) for j, value in enumerate(x_line)]
            line = str(yv[i]) + " " + " ".join(x_line) + "\n"
            f.write(line)
            
if __name__ == "__main__":
    # used by the open-mpi
    # generate("big_size_small_feature", n_samples=1100000, n_features=50, n_informative=10)
    # generate("middle_size_small_feature", n_samples=110000, n_features=50, n_informative=10)
    # generate("small_size_small_feature", n_samples=11000, n_features=50, n_informative=10)
    # generate("tiny_size_small_feature", n_samples=1100, n_features=50, n_informative=10)
    
    # # used by some feature parallel
    # generate("middle_size_middle_feature", n_samples=110000, n_features=100, n_informative=80)
    # generate("middle_size_big_feature", n_samples=110000, n_features=400, n_informative=300)
    generate("middle_size_middle2_feature", n_samples=110000, n_features=500, n_informative=300)
    
    
    