import pandas as pd


def load_dfs(path, frac=0.8):
    """
    :param path: CSV file path containing the data
    :type path: `str`

    :returns: train and test datasets
    :rtype: `tuple[pandas.DataFrame]`
    """
    df = pd.read_csv(path).dropna()
    train = df.sample(frac=frac)
    test = df.drop(train.index)
    return train, test


def random_sample(df, target, shape):
    """
    :param df: dataset to choose random samples from
    :type df: `pandas.DataFrame`
    :param shape: dimensions of the sample
    :type shape: `tuple[int]`
    :param target: series of df with the class to predict
    :type target: `pandas.Series`

    :returns: random subdataset + target dataset
    :rtype: `pandas.DataFrame`
    """
    n_rows, n_cols = shape
    df = df.sample(n=n_cols, axis=1)
    df[target.name] = target
    df = df.sample(n=n_rows)
    return df


if __name__ == "__main__":
    train, test = load_dfs("data.csv")
    print(random_sample(train.drop("Class", axis=1), train["Class"], (1, 1)))
    print(train["Class"])
