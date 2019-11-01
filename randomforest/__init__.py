from randomforest.preprocess import random_sample
from randomforest.decisiontree import DecisionTreeModel

try:
    from progress.bar import Bar
except (ModuleNotFoundError, ImportError):
    class Bar:
        def __init__(self, msg):
            print(msg)

        def iter(self, iterator):
            return iterator


class RandomForestModel:
    def fit(self, df, target_name, n_estimators=5, frac_shape=(0.2, 0.3),
            max_depth=4, min_samples_split=1000):
        """
        :param df: dataset to fit
        :type df: `pandas.DataFrame`
        :param target_name: name of the column to predict
        :type target_name: `str`
        :param max_depth: maximum depth of the tree can have
        :type max_depth: `int`
        :param min_samples_split: minimum number of samples a node can have to split
        :type min_samples_split: `int`

        :returns: does not return
        :rtype: `None`
        """
        if n_estimators < 1:
            raise ValueError('there must be at least one estimator')
        if target_name not in df.columns:
            raise ValueError('the target column must be present in data')
        if df.shape[0] == 0:
            raise ValueError('the data must have at least one row')
        if df.shape[1] < 2:
            raise ValueError('the data must have at least two columns')
        if max_depth < 0:
            raise ValueError('the maximum depth must be non negative')

        target = df[target_name]
        df_exc = df.drop(columns=[target_name])

        n_shape = (
            max(int(frac_shape[0] * df_exc.shape[0]), 1),
            max(int(frac_shape[1] * df_exc.shape[1]), 1),
        )

        self.forest = []
        self.target_name = target_name

        for _ in Bar('Training...').iter(range(n_estimators)):
            sample_df = random_sample(df_exc, target, n_shape)
            tree = DecisionTreeModel()
            tree.fit(sample_df, target_name, max_depth, min_samples_split)
            self.forest.append(tree)

    def predict(self, data):
        """
        :param data: information for predicting
        :param type: `pandas.Series`

        :returns: a random forest prediction
        :rtype: `obj`
        """
        if 'forest' not in self.__dict__:
            raise AttributeError('must fit the model before predicting')

        predictions = []
        for tree in self.forest:
            prediction = tree.predict(data)
            predictions.append(prediction)
        return max(set(predictions), key=predictions.count)

    def assert_predictions(self, df):
        """
        :param df: dataset to test
        :type df: `pandas.DataFrame`

        :returns: correct percentage
        :rtype: `str`
        """
        assertions = []
        for _, data in df.iterrows():
            assertion = self.predict(data) == data[self.target_name]
            assertions.append(assertion)
        return f'{sum(assertions) / len(assertions) * 100}%'
