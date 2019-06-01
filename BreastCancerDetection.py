from math import *
import time

print("Welcome to the breast cancer detection system!")
time.sleep(2)
print ("Using the features of ClumpThickness,UniformityofCellSize,UniformityofCellShape,MarginalAdhesion,SingleEpithelialCellSIze,BareNuclei,BlandChromatin,NormalNucleoli,and Mitoses of both malignant and benign patients, we will predict whether or not a new patient has breast cancer!")
print()
time.sleep(10)
file = input("Which testing file would you like to use for breast cancer detection? ")

def get_data_unlabeled():
    x = []
    input_array = open(file).read().split("\n")
    for index, i in enumerate(input_array):
        split_line = i.split(",")
        if len(split_line) == 9:
            x.append(list(map(int, split_line)))

    return x


def get_data_labeled():
    x = []
    input_array = open("trainingdata.csv").read().split("\n")
    for i in input_array:
        split_line = i.split(",")
        if len(split_line) == 10:
            x.append(list(map(int, split_line)))

    return x


def generate_output_file(y_test):
    with open('BreastCancerPredictions.csv', 'w') as f:
        f.write("Patient #,classification\n")
        for i in range(len(y_test)):
            if (y_test[i] == 0):
               classification = 'Benign'
            else:
               classification = 'Malignant'
            f.write("Patient " + str(i+1)+","+str(classification)+"\n")


train_data = get_data_labeled()
test_features = get_data_unlabeled()

def class_count(data):
    num_benign = 0
    num_malignant = 0
    for data_point in data:
        if data_point[-1] == 0:
            num_benign = num_benign + 1
        else:
            num_malignant = num_malignant + 1
    return num_benign, num_malignant


def gini_impurity(data):
    num_benign, num_malignant = class_count(data)
    return 1 - ((num_benign/len(data)) ** 2 + (num_malignant/len(data)) ** 2)


def entropy(data):
    num_benign, num_malignant = class_count(data)
    if num_malignant == 0 or num_benign == 0:
        return 0
    return -1 * (((num_benign/len(data)) * log2(num_benign/len(data))) + ((num_malignant/len(data)) * log2(num_malignant/len(data))))


def information_gain(data, left, right):
    data_impurity = gini_impurity(data)

    if len(left) == 0:
        left_impurity = 0
    else:
        left_impurity = gini_impurity(left)

    if len(right) == 0:
        right_impurity = 0
    else:
        right_impurity = gini_impurity(right)

    n = len(data)
    n_left = len(left)
    n_right = len(right)
    return data_impurity - (n_left/n) * left_impurity - (n_right/n) * right_impurity


def information_gain_entropy(data, left, right):
    data_impurity = entropy(data)

    if len(left) == 0:
        left_impurity = 0
    else:
        left_impurity = entropy(left)

    if len(right) == 0:
        right_impurity = 0
    else:
        right_impurity = entropy(right)

    n = len(data)
    n_left = len(left)
    n_right = len(right)
    return data_impurity - (n_left / n) * left_impurity - (n_right / n) * right_impurity


def split(data, feature, threshold):
    left = []
    right = []
    for data_point in data:
        if data_point[feature] < threshold:
            left.append(data_point)
        else:
            right.append(data_point)
    return left, right


def best_split(data):
    best_feature = 0
    best_threshold = 1
    ig = -1
    left = []
    right = []
    for feature in range(9):
        for threshold in range(1, 11):
            pot_left, pot_right = split(data, feature, threshold)
            pot_ig = information_gain(data, pot_left, pot_right)
            if pot_ig > ig:
                best_feature = feature
                best_threshold = threshold
                left = pot_left
                right = pot_right
                ig = pot_ig

    return best_feature, best_threshold, ig, left, right


class Leaf:
    def __init__(self, rows):
        num_benign, num_malignant = class_count(rows)
        if num_benign > num_malignant:
            self.prediction = 0
        else:
            self.prediction = 1


class DecisionNode:
    def __init__(self, feature, threshold, left_node, right_node):
        self.feature = feature
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node


def build_tree(data):
    feature, threshold, ig, left, right = best_split(data)

    if ig == 0:
        return Leaf(data)

    left_branch = build_tree(left)
    right_branch = build_tree(right)

    return DecisionNode(feature, threshold, left_branch, right_branch)


def classify(row, node):
    if isinstance(node, Leaf):
        return node.prediction

    if row[node.feature] < node.threshold:
        return classify(row, node.left_node)
    else:
        return classify(row, node.right_node)


print()
print ("We are currently making predictions...")
tree = build_tree(train_data)

time.sleep(5)

y_test = []
for i in range(len(test_features)):
    y_test.append(classify(test_features[i], tree))

generate_output_file(y_test)
print()
print ("The predictions are done! Open up the 'BreastCancerPredictions.csv' file to see the classifications for each patient!")
