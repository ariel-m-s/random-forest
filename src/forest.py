from preprocess import random_sample
from tree import DecisionTreeModel


class RandomForestModel:
    def fit(self, df, target_name, n_estimators=100, frac_shape=(0.2, 0.4),
            max_depth=float('inf'), min_samples=0):
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
        if n_estimators < 1:
            raise ValueError('there must be at least one estimator')
        if target_name not in df.columns:
            raise ValueError('the target column must be present in data')
        if df.shape[0] == 0:
            raise ValueError('the data must have at least one row')
        if max_depth < 0:
            raise ValueError('the maximum depth must be non negative')

        target = df[target_name]
        df_exc = df.drop(columns=[target_name])

        forest = []

        for _ in range(n_estimators):
            sample_df = random_sample(df_exc, target, frac_shape)
            tree = DecisionTreeModel()
            tree.fit(sample_df, target_name, max_depth, min_samples)
            forest.append(tree)
