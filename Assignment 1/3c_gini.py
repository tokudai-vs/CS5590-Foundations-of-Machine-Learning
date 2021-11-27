# FoML Assign 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.

import csv
import math
import numpy as np

# Enter You Name Here
myname = "Vishal"  # or "Amar-Akbar-Antony"

# Implement your decision tree below
class DecisionTree:
    def __init__(self):
        self.tree = {}

    def learn(self, training_set):
        # implement this function
        training_set = np.array(training_set, dtype=float)
        used_feature = []
        self.tree = self.build_tree(
            training_set[:, :-1], training_set[:, -1].astype(int), used_feature
        )
        # print(self.tree)

    def build_tree(self, training_set, target, used_feature):
        # algorithm from https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf

        if len(set(target)) == 1:
            return target[0]

        if (training_set == training_set[0]).all() or len(used_feature) == 11:
            return self.majority_class(target)

        feature_to_split, value_to_split = self.split_feature(
            training_set, target, used_feature
        )

        left_x, right_x, left_y, right_y = self.split(
            training_set, target, feature_to_split, value_to_split
        )

        tree = {}
        tree[str(feature_to_split)] = [
            value_to_split,
            self.build_tree(left_x, left_y, used_feature),
            self.build_tree(right_x, right_y, used_feature),
        ]

        return tree

    def majority_class(self, target):
        # find the class which has majority
        counts = np.bincount(target)
        if counts[0] > counts[1]:
            return 0
        else:
            return 1

    def split_feature(self, training_set, target, used_feature):
        # find best feature to split on and get the best features threshold
        gini = 10000
        feature = -1
        value = -1

        for idx in range(len(training_set[0])):
            if idx in used_feature:
                # already used the feature so don't use it again
                continue

            split_points = self.get_threshold(training_set[:, idx], target)

            for point in split_points:
                left, right = self.divide(training_set[:, idx], target, point)

                divided_target = []
                divided_target.append(left)
                divided_target.append(right)

                gini_new = self.get_gini(len(target), divided_target)

                if gini_new < gini:
                    feature = idx
                    value = point
                    gini = gini_new

        used_feature.append(feature)
        return feature, value

    def get_threshold(self, training_set, target):
        # get possible thresholds for a given feature
        # a possible threshold is one where target value changes
        thresholds = []
        for i in range(len(training_set) - 1):
            if target[i] != target[i + 1]:
                thresholds.append((training_set[i] + training_set[i + 1]) / 2)
        thresholds.sort()
        return thresholds

    def divide(self, training_set, target, point):
        left = []
        right = []
        for i in range(len(training_set)):
            if training_set[i] < point:
                left.append(target[i])
            else:
                right.append(target[i])
        return left, right

    def get_gini(self, num_target, divided_target):
        l = divided_target[0]
        r = divided_target[1]

        l_counts = [np.count_nonzero(l == 0), np.count_nonzero(l == 1)]
        r_counts = [np.count_nonzero(r == 0), np.count_nonzero(r == 1)]

        prob_l = 1 - (
            math.pow(l_counts[0] / len(l), 2) + math.pow(l_counts[1] / len(l), 2)
        )
        prob_r = 1 - (
            math.pow(r_counts[0] / len(l), 2) + math.pow(r_counts[1] / len(l), 2)
        )

        return len(l) / num_target * prob_l + len(r) / num_target * prob_r

    def split(self, training_set, target, feature_to_split, value_to_split):
        # split data based on  a feature
        feature = training_set[:, feature_to_split]

        # np.delete(training_set, feature_to_split, axis =1)

        left_x = []
        right_x = []
        left_y = []
        right_y = []

        for idx in range(len(feature)):
            if feature[idx] < value_to_split:
                left_x.append(training_set[idx])
                left_y.append(target[idx])
            else:
                right_x.append(training_set[idx])
                right_y.append(target[idx])
        left_x = np.array(left_x, dtype=float)
        right_x = np.array(right_x, dtype=float)
        left_y = np.array(left_y, dtype=int)
        right_y = np.array(right_y, dtype=int)

        return left_x, right_x, left_y, right_y

    # implement this function
    def classify(self, test_instance):
        result = self.tree
        while type(result) == dict:
            feature = list(result.keys())[0]
            if float(test_instance[int(feature)]) < result[feature][0]:
                result = result[feature][1]
            else:
                result = result[feature][2]
        return int(result)


def run_decision_tree():

    # Load data set
    with open("wine-dataset.csv") as f:
        next(f, None)
        data = [tuple(line) for line in csv.reader(f, delimiter=",")]
    print("Number of records: %d" % len(data))

    # Split training/test sets
    # You need to modify the following code for cross validation.
    K = 10
    training_set = [x for i, x in enumerate(data) if i % K != 9]
    test_set = [x for i, x in enumerate(data) if i % K == 9]

    tree = DecisionTree()
    # Construct a tree using training set
    tree.learn(training_set)

    # Classify the test set using the tree we just constructed
    results = []
    for instance in test_set:
        result = tree.classify(instance[:-1])
        results.append(result == int(instance[-1]))

    # Accuracy
    accuracy = float(results.count(True)) / float(len(results))
    print("accuracy: %.4f" % accuracy)

    # Writing results to a file (DO NOT CHANGE)
    f = open(myname + "result.txt", "w")
    f.write("accuracy: %.4f" % accuracy)
    f.close()


if __name__ == "__main__":
    run_decision_tree()
