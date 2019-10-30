import numpy as np
from preprocess import load_dfs


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
        if max_depth < 0:
            raise ValueError('cannot construct tree with negative depth')

        self.df = df
        self.target_name = target_name
        self.max_depth = max_depth
        self.min_samples = min_samples


class DecisionTreeNode:
    def __init__(self, df, target, min_rows, max_depth):
        """
        :param df: dataset to process excluding the target
        :type df: `pandas.DataFrame`
        :param target_name: column to predict
        :type target: `pandas.Series`
        :param max_depth: maximum depth that this subtree can have
        :type max_depth: `int`
        :param min_samples: minimum number of samples a node can have to split
        :type min_samples: `int`

        :returns: does not return
        :rtype: `None`
        """
        self.target = target
        self.size = df.shape[0]
        self.select_children(df)

    def group_entropy(self, candidate):
        """
        :param candidate: one possible way to split the dataset
        :type candidate: `tuple[str, dict[string: pandas.DataFrame]]`

        :returns: the weighted average entropy of the children
        :rtype: `float`
        """
        _, group = candidate
        return np.sum((df.shape[0] / self.size) * entropy(df[self.target])
                      for df in group.values())

    def select_children(self, df):
        """
        :param df: parent dataset excluding the target
        :type df: `pandas.DataFrame`

        :returns: the selected way to split the dataset
        :rtype: `tuple[str, dict[string: pandas.DataFrame]]`
        """
        candidates = [(feat_name, {key: value
                                   for key, value in df.groupby(feat_name)})
                      for feat_name in df.columns]
        return min(candidates, key=self.group_entropy)


if __name__ == "__main__":
    train, test = load_dfs('data.csv')
    print(DecisionTreeNode.generate_possible_children(test))
