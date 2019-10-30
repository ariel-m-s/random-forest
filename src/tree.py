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
    return -np.sum(p * np.log2(p))


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
        if df.shape[0] == 0:
            raise ValueError('cannot construct tree with no data')
        if max_depth < 0:
            raise ValueError('cannot construct tree with negative depth')

        self.tree = DecisionTree(
            df, target_name, max_depth, max(min_samples, 2))


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
        certainty = df[target_name].nunique == 1
        self.leaf = (max_depth == 0 and self.size < min_samples) or certainty

        if self.leaf:
            self.df = df
            self.target_name = target_name
            return

        self.split_feat_name, winner = self.split_dataset(df, target_name)
        child_kwargs = {
            'target_name': target_name,
            'max_depth': max_depth - 1,
            'min_samples': min_samples
        }
        self.children = {key: DecisionTree(
            df=value.drop(columns=self.split_feat_name), **child_kwargs)
                          for key, value in winner}

    @property
    def prediction(self):
        if not self.leaf:
            raise AttributeError('a non-leaf node has no prediction to make')
        if '_prediction' in self.__dict__:
            return self._prediction
        target = self.df[self.target_name]
        return most_repeating_value(target)

    def group_entropy(self, dfs, target_name, parent_count):
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
        return np.sum((df.shape[0] / parent_count) * entropy(df[target_name])
                      for df in dfs)

    def split_dataset(self, df, target_name):
        """
        :param df: parent dataset
        :type df: `pandas.DataFrame`

        :returns: the selected way to split the dataset
        :rtype: `tuple[str, dict[string: pandas.DataFrame]]`
        """
        candidates = [(feat_name, {key: value
                                   for key, value in df.groupby(feat_name)})
                      for feat_name in df.columns]
        parent_count = df.shape[0]
        return min(candidates, key=lambda candidate: self.group_entropy(
            candidate[1].values(), target_name, parent_count))


if __name__ == "__main__":
    # train, test = load_dfs('data.csv')
    # print(DecisionTreeNode.generate_possible_children(test))
    import pandas as pd
    most_repeating_value(pd.Series([]))
