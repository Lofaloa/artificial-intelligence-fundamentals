from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn import datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    X, y = make_moons(n_samples=10000, noise=0.1, random_state=42)
    moons_df = pd.DataFrame(data = X)
    moons_df.plot.scatter(x = 0, y = 1, c = y, colormap="jet")
    plt.show()

if __name__ == "__main__":
    main()