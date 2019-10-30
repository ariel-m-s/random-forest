import numpy as np


def entropy(feat):
    """
    :param feat: column to which to calculate the entropy
    :type feat: `pandas.Series`

    :returns: entropy of the column
    :rtype: `float`
    """
    _, counts = np.unique(feat, return_counts=True)
    p = counts / feat.count()
    return -sum(p * np.log2(p))


def group_entropy(dfs, target_name, parent_count):
    """
    :param dfs: one possible way to split the dataset
    :type dfs: `dict_values[pandas.DataFrame]`
    :param target_name: name of the column to predict
    :type target_name: `str`
    :param parent_count: the number of rows in the parent dataset
    :type parent_count: `int`

    :returns: the weighted average entropy of the children
    :rtype: `float`
    """
    return sum((df.shape[0] / parent_count) * entropy(df[target_name])
               for df in dfs)


def most_repeating_value(feat):
    """
    :param feat: column to which to calculate the MRV
    :type feat: `pandas.Series`

    :returns: MRV
    :rtype: `obj`
    """
    _, counts = np.unique(feat, return_counts=True)
    index = np.argmax(counts)
    return feat[index]


class DecisionTreeModel:
    def fit(self, df, target_name, max_depth, min_samples):
        """
        :param df: dataset to fit
        :type df: `pandas.DataFrame`
        :param target_name: name of the column to predict
        :type target_name: `str`
        :param max_depth: maximum depth of the tree can have
        :type max_depth: `int`
        :param min_samples: minimum number of samples a node can have to split
        :type min_samples: `int`

        :returns: does not return
        :rtype: `None`
        """
        if target_name not in df.columns:
            raise ValueError('the target column must be present in data')
        if df.shape[0] == 0:
            raise ValueError('the data must have at reast one row')
        if max_depth < 0:
            raise ValueError('the maximum depth must be non negative')

        self.tree = DecisionTree(df, target_name, max_depth, min_samples)

    def predict(self, data):
        if 'tree' not in self.__dict__:
            raise AttributeError('must fit the model before predicting')

        tree = self.tree
        while not tree.leaf:
            value = data[tree.split_feat_name]
            tree = tree.children.get(value, tuple(tree.children.values())[0])
        return tree.prediction


class DecisionTree:
    def __init__(self, df, target_name, max_depth, min_samples):
        """
        :param df: dataset to process
        :type df: `pandas.DataFrame`
        :param target_name: name of the column to predict
        :type target_name: `str`
        :param max_depth: maximum depth that this subtree can have
        :type max_depth: `int`
        :param min_samples: minimum number of samples a node can have to split
        :type min_samples: `int`

        :returns: does not return
        :rtype: `None`
        """
        self.leaf = (max_depth == 0 and df.shape[0] < min_samples) \
            or df[target_name].nunique == 1

        if self.leaf:
            self.df = df
            target = self.df[self.target_name]
            self.prediction = most_repeating_value(target)

        else:
            self.split_feat_name, winner = self.split_dataset(df, target_name)
            child_kwargs = {
                'target_name': target_name,
                'max_depth': max_depth - 1,
                'min_samples': min_samples
            }
            self.children = {key: DecisionTree(
                df=value.drop(columns=self.split_feat_name), **child_kwargs)
                              for key, value in winner.items()}

    def split_dataset(self, df, target_name):
        """
        :param df: parent dataset
        :type df: `pandas.DataFrame`

        :returns: the selected way to split the dataset
        :rtype: `tuple[str, dict[string: pandas.DataFrame]]`
        """
        candidates = [(feat_name, {key: value
                                   for key, value in df.groupby(feat_name)})
                      for feat_name in set(df.columns) - {target_name}]
        parent_count = df.shape[0]
        return min(candidates, key=lambda candidate: group_entropy(
            candidate[1].values(), target_name, parent_count))


if __name__ == "__main__":
    from preprocess import load_dfs
    train, test = load_dfs('mini.csv')
    model = DecisionTreeModel()
    model.fit(train, 'Class', 5, 10)
