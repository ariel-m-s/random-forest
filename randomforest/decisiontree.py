import numpy as np
import plotly.graph_objects as go


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
    :type dfs: `dict_values`
    :param target_name: name of the column to predict
    :type target_name: `str`
    :param parent_count: the number of rows in the parent dataset
    :type parent_count: `int`

    :returns: weighted average entropy of the children
    :rtype: `float`
    """
    return sum((df.shape[0] / parent_count) * entropy(df[target_name])
               for df in dfs)


def candidates(df, target_name):
    """
    :param df: parent dataset to split
    :type df: `pandas.DataFrame`
    :param target_name: name of the column to predict
    :type target_name: `str`

    :returns: the list of possible ways to split the dataset
    :rtype: `list`
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
    def fit(self, df, target_name, max_depth=float('inf'), min_samples=0):
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
        self.tree = DecisionTree(df, target_name, max_depth, min_samples)

    def predict(self, data):
        """
        :param data: information for predicting
        :param type: `pandas.Series`

        :returns: a decision tree prediction
        :rtype: `obj`
        """
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

    def generate_treemap(self):
        return self.tree.generate_treemap()

    def __str__(self):
        """
        :returns: a visual representation of the decision tree
        :rtype: `str`
        """
        return str(self.tree)


class DecisionTree:
    counter = 0

    def __init__(self, df, target_name, max_depth, min_samples, depth=0):
        """
        :param df: dataset to process
        :type df: `pandas.DataFrame`
        :param target_name: name of the column to predict
        :type target_name: `str`
        :param max_depth: maximum depth that this subtree can have
        :type max_depth: `int`
        :param min_samples: minimum number of samples a node can have to split
        :type min_samples: `int`
        :param depth: depth of this subtree relative to the root node
        :type depth: `int`

        :returns: does not return
        :rtype: `None`
        """
        DecisionTree.counter += 1
        self.id = DecisionTree.counter

        self.depth = depth

        self.leaf = max_depth == 0 \
            or df.shape[0] < min_samples \
            or df.shape[1] <= 1 \
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
                'min_samples': min_samples,
                'depth': depth + 1,
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
        :rtype: `tuple`
        """
        parent_count = df.shape[0]
        return min(candidates(df, target_name),
                   key=lambda candidate: group_entropy(
                       candidate[1].values(), target_name, parent_count))

    def label(self):
        return (f'{self.split_feat_name} {"{}"} {self.split_feat_median}'
                f' (node id: {self.id})')

    def generate_treemap(self, parent_label='', labels=[], parents=[]):
        if self.leaf:
            parents.append(parent_label)
            labels.append((f'{self.prediction} (node id: {self.id})'))

        else:
            for key, child in self.children.items():
                parents.append(parent_label)
                child_label = self.label().format('>' if key else '<=')
                labels.append(child_label)
                child.generate_treemap(child_label, labels, parents)

        if parent_label == '':
            return go.Figure(go.Treemap(labels=labels, parents=parents))

    def __str__(self):
        """
        :returns: a recursive visual representation of the decision tree
        :rtype: `str`
        """
        padding = '   '

        if self.leaf:
            return (f'{padding * self.depth}L{self.depth} - '
                    f'prediction: "{self.prediction}"!\n')

        return (f'{padding * self.depth}L{self.depth} - '
                f'"{self.split_feat_name}" <= {self.split_feat_median}:\n'

                f'\n{self.children.get(False, "empty tree")}\n'

                f'{padding * self.depth}L{self.depth} - '
                f'"{self.split_feat_name}" > {self.split_feat_median}:\n'

                f'\n{self.children.get(True, "empty tree")}\n')
