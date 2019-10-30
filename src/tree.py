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





if __name__ == "__main__":
    train, test = load_dfs('data.csv')
    print(DecisionTreeNode.generate_possible_children(test))
