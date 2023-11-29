import pandas as pd
import torch.nn as nn


class Model(nn.modules):
    def __init__(self):
        self.conv1 = nn.Conv2d()
        self.activation = nn.ReLU()

        pass

    def forward(self, X):
        x = self.conv1(X)
        outputs = self.activation(x)
        return outputs


def main():
    path = ""
    df = pd.read_csv(path)
    x_train = df["features"]
    y_train = df["target"]
    model = Model()


if __name__ == "__main__":
    main()
