import pandas as pd


def load_dfs(path, frac=0.8):
    """
    path: `str`

    return: `tuple[DataFrame]`
    """
    df = pd.read_csv(path).dropna()
    train = df.sample(frac=frac)
    test = df.drop(train.index)
    return train, test


if __name__ == "__main__":
    train, test = load_dfs("data.csv")
    print(train.head())
    print(test.head())
