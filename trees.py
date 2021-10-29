# Code for partial fulfillment of the 2021/22 Introduction to Machine Learning course
# taught by the Department of Computing at Imperial College London.

# Contributors:
#   Aniruddha Hazarika
#   Thomas Lewis
#   Igor Silin
#   Varun Srivastava

# ----------------------------------------------------------------------------------- #
import numpy as np
import matplotlib as mlp
mlp.use('gtk3agg')
import matplotlib.pyplot as plt
import math

"""
split an array based on a threshold in a selected column

arr: array to split
split_name: column which is being compared
threshold: the threshold to split the array
"""
def split(arr, split_name, threshold):
    left_split = arr[arr[:,split_name] < threshold]
    right_split = arr[arr[:,split_name] >= threshold]
    
    return left_split,right_split


"""
calculcate the entropy of the column passed in

column: a list of the rooms numbers in the current array being checked
"""
def calc_entropy(column):
    unique, counts = np.unique(column, return_counts=True) #get a list of the unique values in the array and their number of occurrences
    dict(zip(unique, counts))
    probability = counts / len(column)

    entropy = 0

    for prob in probability: #entropy calculation
        if prob>0:
            entropy += prob*math.log(prob,2)

    return -entropy


"""
calculate the information gain from the chosen split column and threshold

data: the current dataset
split_name: current column to split on
target_name: the column of rooms number
threshold: the threshold to split the array
"""
def calc_information_gain(data, split_name, target_name, threshold):
    original_entropy = calc_entropy(data[:,target_name]) #find the entropy without the split

    left_split,right_split = split(data,split_name,threshold) #split the array

    to_subtract = 0

    for subset in [left_split, right_split]:
        prob = (subset.shape[0]/data.shape[0])
        to_subtract += prob*calc_entropy(subset[:,target_name]) #find the entropies of the split arrays

    return original_entropy - to_subtract #return information gain


"""
find the highest information gain from all the possible split columns and thresholds

dataset: current dataset
target_column: the column of room numbers
"""
def highest_information_gain(dataset, target_column):
    information_gains = np.array([0,0,0])

    for col in range(7):
        ordered_dataset = dataset[np.argsort(dataset[:,col])] #sort the dataset based on the current column

        unique_vals = np.unique(ordered_dataset[:,col]) #get all the unique values in the current column
        
        for i in range(unique_vals.shape[0]-1):
            first_val = unique_vals[i]
            second_val = unique_vals[i+1]
            threshold = (first_val+second_val)/2 #find the current threshold
        
            information_gain = calc_information_gain(dataset,col,target_column,threshold)
            information_gains = np.vstack((information_gains,np.array([col,threshold,information_gain])))
                
    max_index = np.argmax(information_gains[:,2]) #find the max information gain
    
    return information_gains[max_index]


"""
find the optimum split column and threshold

dataset: the current dataset
"""
def find_split(dataset):
    target_column = 7
    return highest_information_gain(dataset,target_column)

# Returns the tree (can either use a tree class or nested dictionaries)
def decision_tree_learning(training_dataset, depth):
    unique_vals = np.unique(training_dataset[:,7])
    if unique_vals.size == 1:
        return ({"leaf": True, "room": int(unique_vals[0])},depth)
    else:
        split_point = find_split(training_dataset)
        l_dataset, r_dataset = split(training_dataset,int(split_point[0]),split_point[1])
        l_branch, l_depth = decision_tree_learning(l_dataset,depth+1)
        r_branch, r_depth = decision_tree_learning(r_dataset,depth+1)
        node = {"leaf": False, "attribute": int(split_point[0]), "value": split_point[1], "left": l_branch, "right": r_branch}
        return (node, max(l_depth,r_depth))

def classify(data, trained_tree):
    if trained_tree["leaf"]:
        return trained_tree["room"]
    else:
        if data[trained_tree["attribute"]] < trained_tree["value"]:
            return classify(data, trained_tree["left"])
        else:
            return classify(data, trained_tree["right"])

def evaluate(test_db, trained_tree):
    correct = 0
    for i in range(test_db.shape[0]):
        if test_db[i,7] == classify(test_db[i,0:7],trained_tree):
            correct += 1
    return correct/test_db.shape[0]

def plot_node(target_node,x,y):
  max_width = 100
  height = 5
  depth=abs(y)/height
  if target_node['leaf'] == False:
      plot_node(target_node['left'],(x-(max_width)/(2**depth)),y-height)
      plot_node(target_node['right'],(x+(max_width)/(2**depth)),y-height)
      plt.plot([(x-(max_width)/(2**depth)),x,(x+(max_width)/(2**depth))],[y-height,y,y-height])
      title = "[X%d < %4.2f]"%(target_node['attribute'],target_node['value'])
  else:
    title="Room: %d" %target_node['room']
  plt.text(x, y, title, ha="center", va="center",
            size=5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="b", lw=2))
    
  plt.axis('off')  
    
    
    
    
#Training the tree and testing it
dataset = np.loadtxt("clean_dataset.txt")
tree, depth = decision_tree_learning(dataset,0)
testset = np.loadtxt("noisy_dataset.txt")
print(evaluate(testset,tree))
plot_node(tree,0,0)
plt.show()