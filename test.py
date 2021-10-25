import numpy as np
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


dataset = np.loadtxt("clean_dataset.txt")
#print(dataset[:5,:])


print(find_split(dataset))
