# Code for partial fulfillment of the 2021/22 Introduction to Machine Learning course
# taught by the Department of Computing at Imperial College London.

# Contributors:
#   Aniruddha Hazarika
#   Thomas Lewis
#   Igor Silin
#   Varun Srivastava

# ----------------------------------------------------------------------------------- #

import numpy as np
from numpy.random import default_rng
#Uncomment the next line 
#import matplotlib.pyplot as plt
import math

def pretty_print_depth(cm,recall,precision,f1,accuracy,unpruned_depth,pruned_depth):
    pretty_print(cm,recall,precision,f1,accuracy)
    print("\nAverage Depth")
    print("Unpruned: " + str(round(unpruned_depth,2)))
    print("Pruned: " + str(round(pruned_depth,2)))

def pretty_print(cm,recall,precision,f1,accuracy):
    print("Confusion Matrix")
    print(cm)
    print("\nRecall")
    print("Room 1: " + str(round(recall[0],3)))
    print("Room 2: " + str(round(recall[1],3)))
    print("Room 3: " + str(round(recall[2],3)))
    print("Room 4: " + str(round(recall[3],3)))
    print("\nPrecision")
    print("Room 1: " + str(round(precision[0],3)))
    print("Room 2: " + str(round(precision[1],3)))
    print("Room 3: " + str(round(precision[2],3)))
    print("Room 4: " + str(round(precision[3],3)))
    print("\nF1")
    print("Room 1: " + str(round(f1[0],3)))
    print("Room 2: " + str(round(f1[1],3)))
    print("Room 3: " + str(round(f1[2],3)))
    print("Room 4: " + str(round(f1[3],3)))
    print("\nAccuracy")
    print(str(round(accuracy,3)))

"""
split an array based on a threshold in a selected column

arr: array to split
split_name: column which is being compared
threshold: the threshold to split the array
"""
def split(arr, split_name, threshold):
    if arr.size == 0:
        return np.array([]), np.array([])
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
        return int(trained_tree["room"])
    else:
        if data[trained_tree["attribute"]] < trained_tree["value"]:
            return classify(data, trained_tree["left"])
        else:
            return classify(data, trained_tree["right"])

#Evaluation function that returns accuracy
def evaluate(test_db, trained_tree):
    correct = 0
    for i in range(test_db.shape[0]):
        if test_db[i,7] == classify(test_db[i,0:7],trained_tree):
            correct += 1
    return correct/test_db.shape[0]

#Evaluation function that returns the confusion matrix
def evaluate_cm(test_db, trained_tree):
    cm = np.zeros((4,4))
    for i in range(test_db.shape[0]):
        room = classify(test_db[i,0:7],trained_tree)
        gold = int(test_db[i,7])
        cm[gold-1,room-1] += 1
    return cm

def calc_metrics(cm):
    recall = np.zeros(4)
    precision = np.zeros(4)
    f1 = np.zeros(4)
    accuracy = (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3])/cm.sum()
    cols = cm.sum(axis=0)
    rows = cm.sum(axis=1)
    for i in range(4):
        precision[i] = cm[i,i] / cols[i]
        recall[i] = cm[i,i] / rows[i]
        f1[i] = (2*precision[i]*recall[i])/(precision[i]+recall[i])
    return recall, precision, f1, accuracy

#Evaluation function that returns the confusion matrix, recall, precision, f1 and accuracy
def evaluate_plus(test_db, trained_tree):
    cm = evaluate_cm(test_db, trained_tree)
    return (cm,) + calc_metrics(cm)


def k_fold_split(n_splits,n_instances,random_generator=default_rng()):
    shuffled_indices = random_generator.permutation(n_instances)
    split_indices = np.array_split(shuffled_indices,n_splits)
    return split_indices


def train_test_k_fold(n_folds,n_instances,random_generator=default_rng()):
    split_indices = k_fold_split(n_folds,n_instances,random_generator)

    folds = []
    for k in range(n_folds):
        test_indices = split_indices[k]
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])
        folds.append([train_indices,test_indices])
    return folds

#Returns the confusion matrix, metrics and average depth
def cross_validation(dataset,n_folds):
    dataset = np.loadtxt(dataset)
    np.random.shuffle(dataset)
    cm = np.zeros((4,4))
    for i,(train_indices,test_indices)in enumerate(train_test_k_fold(n_folds,len(dataset))):
        train = dataset[train_indices,:]
        test = dataset[test_indices,:]
        tree,depth = decision_tree_learning(train,0)
        cm += evaluate_cm(test,tree)
    metrics = calc_metrics(cm)
    return (cm/n_folds,) + metrics

def majority(vals,counts):
    max_count = max(counts)
    rooms = []
    for i,item in enumerate(vals):
        if counts[i] == max_count:
            rooms.append(item)
    return rooms


def plot_node(target_node,x,y):
    max_width = 200
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
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="b", lw=0.5))
    
    plt.axis('off')  


# Returns the pruned tree, a bool indicating whether anything changed, and the maximum depth
def prune(tree,training_set,validation_set,depth):
    if tree["leaf"]:
        return tree, False, depth
    elif tree["left"]["leaf"] and tree["right"]["leaf"]:
    # Do pruning in here:
        unique_vals, counts = np.unique(training_set[:,7], return_counts=True)
        majority_rooms = majority(unique_vals,counts)
        if validation_set.size == 0:
            new_correct = 1
            old_correct = 0
            room = majority_rooms[0]
        else:
            v_vals, v_counts = np.unique(validation_set[:,7], return_counts=True)
            v_dict = dict(zip(v_vals,v_counts))
            l_val, r_val = split(validation_set,tree["attribute"],tree["value"])
            l_unique, l_counts = np.unique(l_val[:,7], return_counts=True)
            r_unique, r_counts = np.unique(r_val[:,7], return_counts=True)
            l_dict = dict(zip(l_unique,l_counts))
            r_dict = dict(zip(r_unique,r_counts))
            if tree["left"]["room"] in l_dict:
                l_correct = l_dict[tree["left"]["room"]]
            else:
                l_correct = 0
            if tree["right"]["room"] in r_dict:
                r_correct = r_dict[tree["right"]["room"]]
            else:
                r_correct = 0
            old_correct = l_correct + r_correct 
            #Calculate new accuracy
            new_correct = 0
            max_correct = 0
            best_room = 0
            for r in majority_rooms:
                if r in v_dict:
                    new_correct = v_dict[r]
                    if new_correct > max_correct:
                        max_correct = new_correct
                        best_room = r
            new_correct = max_correct
            room = best_room
        #   Compare accuracy using majority label vs existing decision
        if new_correct >= old_correct:
            return {"leaf": True, "room": room}, True, depth
        else:
            return tree, False, depth+1 #leave it unchanged
    else:
        l_train, r_train = split(training_set,tree["attribute"],tree["value"])
        l_val, r_val = split(validation_set,tree["attribute"],tree["value"])
        left_tree, cl, dl = prune(tree["left"], l_train, l_val, depth+1)
        right_tree, cr, dr = prune(tree["right"], r_train, r_val, depth+1)
        return {"leaf": False, "attribute": tree["attribute"], "value": tree["value"], "left": left_tree, "right": right_tree}, cl or cr, max(dl,dr)

# Returns confusion matrix, average unpruned depth and average pruned depth
def internal_validation(dataset,test_set,n_folds):
    cm = np.zeros((4,4))
    unpruned_depth = 0
    pruned_depth = 0
    for i,(train_indices,val_indices)in enumerate(train_test_k_fold(n_folds,len(dataset))):
        train = dataset[train_indices,:]
        val = dataset[val_indices,:]
        unpruned_tree,depth = decision_tree_learning(train,0)
        #plt.figure(figsize=(20, 10))
        #plot_node(unpruned_tree,0,0)
        #plt.savefig('unpruned.pdf')
        unpruned_depth += depth
        changed = True
        while changed:
            pruned_tree, changed, depth = prune(unpruned_tree,train,val,0)
            unpruned_tree = pruned_tree
        #plt.figure(figsize=(20, 10))
        #plot_node(pruned_tree,0,0)
        #plt.savefig('pruned.pdf')
        cm += evaluate_cm(test_set,pruned_tree)
        pruned_depth += depth
    return cm/n_folds, unpruned_depth/n_folds, pruned_depth/n_folds
        
# Returns confusion matrix, metrics, average unpruned depth and average pruned depth
def nested_cross_validation(dataset,n_folds):
    dataset = np.loadtxt(dataset)
    np.random.shuffle(dataset)
    cm = np.zeros((4,4))
    unpruned_depth = 0
    pruned_depth = 0
    for i,(train_indices,test_indices)in enumerate(train_test_k_fold(n_folds,len(dataset))):
        train = dataset[train_indices,:]
        test = dataset[test_indices,:]
        matrix, u_depth, p_depth = internal_validation(train,test,n_folds)
        cm += matrix
        unpruned_depth += u_depth
        pruned_depth += p_depth
    metrics = calc_metrics(cm)
    return (cm/n_folds,) + metrics + (unpruned_depth/n_folds,pruned_depth/n_folds)

#Simple cross-validation
cm, recall, precision, f1, accuracy = cross_validation("clean_dataset.txt",10)
pretty_print(cm,recall,precision,f1,accuracy)

'''
#Uncomment for nested cross-validation
cm, recall, precision, f1, accuracy, unpruned_depth, pruned_depth = nested_cross_validation("clean_dataset.txt",10)
pretty_print_depth(cm,recall,precision,f1,accuracy,unpruned_depth,pruned_depth)
'''

'''
#Uncomment for tree visualisation
dataset = np.loadtxt("clean_dataset.txt")
tree, depth = decision_tree_learning(dataset,0)
plt.figure(figsize=(20, 10))
plot_node(tree,0,0)
plt.savefig('tree.pdf')
'''