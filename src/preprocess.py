import pandas as pd


def load_dfs(path, frac=0.8):
    """
    :param path: CSV file path containing the data
    :type path: `str`

    :returns: train and test datasets
    :rtype: `tuple[pandas.DataFrame, pandas.DataFrame]`
    """
    df = pd.read_csv(path).dropna()
    train = df.sample(frac=frac)
    test = df.drop(train.index)
    return train, test


def random_sample(df, target, shape):
    """
    :param df: dataset to choose a random sample from
    :type df: `pandas.DataFrame`
    :param shape: dimensions of the sample
    :type shape: `tuple[int, int]`
    :param target: column to predict
    :type target: `pandas.Series`

    :returns: random subdataset + target dataset
    :rtype: `pandas.DataFrame`
    """
    n_rows, n_cols = shape
    df = df.sample(n=n_cols, axis=1)
    df[target.name] = target
    df = df.sample(n=n_rows)
    return df
