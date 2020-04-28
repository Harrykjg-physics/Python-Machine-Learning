from math import sqrt
from random import seed
from random import randrange
from csv import reader


# Load a CSV file


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


#  Convert string column to integer

def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Find the Min Max value for each column

def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0 - 1

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / \
                     (minmax[i][1] - minmax[i][0])


# Split a dataset into k folds

def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate Accuracy percentage

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split

def evaluate_algorithm(dataset, algorithm,
                       n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set,
                              *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# dataset for testing

dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]


# Calculate Euclidean_distance

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# Test distance function

# print("Test distance function")
# row0 = dataset[0]
# for row in dataset:
#     distance = euclidean_distance(row0, row)
#     print(distance)

# get the nearest neighbours


def get_neighbors(train, test_row, num_neighbours):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(train_row, test_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbours):
        neighbors.append(distances[i][0])
    return neighbors


# Test nearest function

# print("Test nearest function")
# neighbors = get_neighbors(dataset, dataset[0], 3)
# for neighbor in neighbors:
#     print(neighbor)

# Make predictions


def predict_classification(train, test_row, num_neighbours):
    neighbors = get_neighbors(train, test_row, num_neighbours)
    output_values = [i[-1] for i in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# Test Prediction function

# print("Test Predictions")
# pred = predict_classification(dataset, dataset[0], 3)
# print("Expect:%d, Got:%d" % (dataset[0][-1], pred))

# KNN algorithm

def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return predictions


# Test KNN on the Iris Datasets

seed(1)
filename = 'iris.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)

# Convert class column to integers

str_column_to_int(dataset, len(dataset[0])-1)

# Evaluate

n_folds = 5
num_neighbors = 5
scores = evaluate_algorithm(dataset,
                            k_nearest_neighbors,
                            n_folds, num_neighbors)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' %
      (sum(scores)/float(len(scores))))


