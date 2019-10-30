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


def candidates(df, target_name):
    """
    :param df: the parent dataset to split
    :type df: `pandas.DataFrame`
    :param target_name: name of the column to predict
    :type target_name: `str`

    :returns: the list of possible ways to split the dataset
    :rtype: `list[tuple[tuple[str, float], dict[bool: pandas.DataFrame]]]
    """
    res = []
    for feat_name in set(df.columns) - {target_name}:
        feat = df[feat_name]
        feat_median = feat.median()
        feat_mask = feat > feat_median
        res.append(((feat_name, feat_median),
                    {key: value for key, value in df.groupby(feat_mask)}))
    return res


def most_repeating_value(feat):
    """
    :param feat: column to which to calculate the MRV
    :type feat: `pandas.Series`

    :returns: MRV
    :rtype: `obj`
    """
    _, counts = np.unique(feat, return_counts=True)
    index = np.argmax(counts)
    return feat.values[index]


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
            default_tree = tuple(tree.children.values())[0]
            try:
                value = data[tree.split_feat_name]
            except KeyError:
                tree = default_tree
            else:
                decision = value > tree.split_feat_median
                tree = tree.children.get(decision, default_tree)
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
        self.leaf = max_depth == 0 \
            or df.shape[0] < min_samples \
            or df[target_name].nunique == 1

        if self.leaf:
            target = df[target_name]
            self.prediction = most_repeating_value(target)

        else:
            split_feat_values, winner = self.split_dataset(df, target_name)
            self.split_feat_name, self.split_feat_median = split_feat_values
            child_kwargs = {
                'target_name': target_name,
                'max_depth': max_depth - 1,
                'min_samples': min_samples
            }
            self.children = {key: DecisionTree(
                df=value.drop(columns=[self.split_feat_name]), **child_kwargs)
                              for key, value in winner.items()}

    def split_dataset(self, df, target_name):
        """
        :param df: parent dataset to split
        :type df: `pandas.DataFrame`
        :param target_name: name of the column to predict
        :type target_name: `str`

        :returns: the selected way to split the dataset
        :rtype: `tuple[tuple[str, float], dict[bool: pandas.DataFrame]]`
        """
        parent_count = df.shape[0]
        return min(candidates(df, target_name),
                   key=lambda candidate: group_entropy(
                       candidate[1].values(), target_name, parent_count))


if __name__ == "__main__":
    from preprocess import load_dfs
    train, test = load_dfs('mini.csv')
    model = DecisionTreeModel()
    model.fit(train, 'Class', 1, 10)
