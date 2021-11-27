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
        visited = []
        self.tree = self.build_tree(
            training_set[:, :-1], training_set[:, -1].astype(int), visited
        )
        # print(self.tree)

    def build_tree(self, training_set, target, visited):
        if len(set(target)) == 1:
            return target[0]

        if (training_set == training_set[0]).all() or len(visited) == 11:
            return self.majority_class(target)

        feature_to_split, value_to_split = self.split_feature(
            training_set, target, visited
        )

        left_x, right_x, left_y, right_y = self.split(
            training_set, target, feature_to_split, value_to_split
        )

        tree = {}
        tree[str(feature_to_split)] = [
            value_to_split,
            self.build_tree(left_x, left_y, visited),
            self.build_tree(right_x, right_y, visited),
        ]

        return tree

    def majority_class(self, target):
        counts = np.bincount(target)
        if counts[0] > counts[1]:
            return 0
        else:
            return 1

    def split_feature(self, training_set, target, visited):
        info_gain = -1
        feature = -1
        value = -1

        for idx in range(len(training_set[0])):
            if idx in visited:
                continue

            split_points = self.get_threshold(training_set[:, idx], target)

            for point in split_points:
                left, right = self.divide(training_set[:, idx], target, point)

                divided_target = []
                divided_target.append(left)
                divided_target.append(right)

                gain = self.information_gain(target, divided_target)

                if gain > info_gain:
                    feature = idx
                    value = point
                    info_gain = gain

        visited.append(feature)
        return feature, value

    def get_threshold(self, training_set, target):
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

    def information_gain(self, target, divided_target):
        parent_entropy = self.entropy(target)
        current_entropy = (
            float(len(divided_target[0]))
            / len(target)
            * self.entropy(divided_target[0])
        ) + (
            float(len(divided_target[1]))
            / len(target)
            * self.entropy(divided_target[1])
        )
        return parent_entropy - current_entropy

    def entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        entropy = 0

        for prob in probabilities:
            if prob > 0:
                entropy += prob * math.log(prob, 2)

        return -entropy

    def split(self, training_set, target, feature_to_split, value_to_split):
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
    k_folds = 10

    k, m = divmod(len(data), k_folds)
    split_data = [
        data[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(k_folds)
    ]

    # print(len(split_data))

    K_results = []
    for i in range(k_folds):
        training_set = []
        for j in range(k_folds):
            if j == i:
                continue
            training_set.extend(split_data[i])

        test_set = []
        test_set.extend(split_data[i])

        tree = DecisionTree()

        tree.learn(training_set)
        results = []
        for instance in test_set:
            result = tree.classify(instance[:-1])
            results.append(result == int(instance[-1]))

        # Accuracy
        accuracy = float(results.count(True)) / float(len(results))

        K_results.append(accuracy)

    print("accuracy: %.4f" % float(np.sum(K_results) / k_folds))

    # Writing results to a file (DO NOT CHANGE)
    f = open(myname + "result.txt", "w")
    f.write("accuracy: %.4f" % float(np.sum(K_results) / k_folds))
    f.close()


if __name__ == "__main__":
    run_decision_tree()
