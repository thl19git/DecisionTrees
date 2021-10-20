# Code for partial fulfillment of the 2021/22 Introduction to Machine Learning course
# taught by the Department of Computing at Imperial College London.

# Contributors:
#   Aniruddha Hazarika
#   Thomas Lewis
#   Igor Silin
#   Varun Srivastava

# ----------------------------------------------------------------------------------- #

import numpy as np
import matplotlib as plt

# Finds the optimal split by maximising information gain
def find_split(training_dataset):
    pass

# Returns the tree (can either use a tree class or nested dictionaries)
def decision_tree_learning(training_dataset, depth):
    pass

# Load the training data (Step 1)
data = np.loadtxt("./clean_dataset.txt")
# Checking data was loaded correctly
print(data[:10])