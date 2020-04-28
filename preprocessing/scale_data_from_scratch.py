from csv import reader

# Load a CSV file
from math import sqrt


def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset


# Convert string column to float

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


"""
Normalization can refer to different techniques
depending on context.

Here, we use normalization to refer to rescaling an input variable
to the range between 0 and 1.

Normalization requires that you know
the minimum and maximum values for each attribute.

The snippet of code below defines the dataset_minmax() function
that calculates the min and max value for each attribute in a
dataset, then returns an array of these minimum and maximum values.

"""


# Find the min and max values for each column

def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Contrive small dataset

dataset = [[50, 30], [20, 90]]
print(dataset)

# Calculate min and max for each column

minmax = dataset_minmax(dataset)
print(minmax)

"""
The calculation to normalize a single value for a column is:
scaled_value = (value - min) / (max - min)
"""


# Rescale dataset columns to the range 0-1

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / \
                     (minmax[i][1] - minmax[i][0])


normalize_dataset(dataset, minmax)
print(dataset)

# Load pima-indians-diabetes dataset

filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)

# convert string columns to float

for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
print(dataset[0])

# Calculate min and max for each column

minmax = dataset_minmax(dataset)

# Normalize columns

normalize_dataset(dataset, minmax)
print(dataset[0])

"""
Standardization is a rescaling technique that refers to 
centering the distribution of the data on the value 0 and 
the standard deviation to the value 1.

Together, the mean and the standard deviation can be used to 
summarize a normal distribution, also called the Gaussian 
distribution or bell curve.

"""


# calculate column means

def column_means(dataset):
    means = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        means[i] = sum(col_values) / float(len(dataset))
    return means


# calculate column standard deviations

def column_stdevs(dataset, means):
    stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        variance = [pow(row[i] - means[i], 2) for row in dataset]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(x / (float(len(dataset) - 1))) for x in stdevs]
    return stdevs


# standardize dataset

def standardize_dataset(dataset, means, stdevs):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - means[i]) / stdevs[i]


# Standardize dataset

dataset = [[50, 30], [20, 90], [30, 50]]
print(dataset)

# Estimate mean and standard deviation

means = column_means(dataset)
stdevs = column_stdevs(dataset, means)
print(means)
print(stdevs)

# standardize dataset

standardize_dataset(dataset, means, stdevs)
print(dataset)
