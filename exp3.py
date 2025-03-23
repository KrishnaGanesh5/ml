import pandas as pd 
import math 
def id3(df, target_attribute_name, attribute_names, default_class=None): 
    # Base cases for recursion 
    # If all instances have the same class, return that class 
    if len(set(df[target_attribute_name])) == 1: 
        return df[target_attribute_name].iloc[0] 
    # If attribute_names is empty, return the default class 
    elif len(attribute_names) == 0: 
        return default_class 
    else: 
        # Calculate information gain for each attribute 
        gains = {attribute_name: information_gain(df, attribute_name, target_attribute_name) for 
attribute_name in attribute_names} 
        # Choose the attribute with the highest information gain 
        best_attribute = max(gains, key=gains.get) 
        # Create an empty tree 
        tree = {best_attribute: {}} 
        # Remove the best attribute from the list of attributes 
        remaining_attributes = [attr for attr in attribute_names if attr != best_attribute] 
        # Recursively build the tree for each value of the best attribute 
        for value in df[best_attribute].unique(): 
            subset = df[df[best_attribute] == value] 
            subtree = id3(subset, target_attribute_name, remaining_attributes, default_class) 
            tree[best_attribute][value] = subtree 
        return tree 
# Define functions for entropy and information gain 
def entropy(probs): 
    return sum([-prob * math.log(prob, 2) for prob in probs if prob != 0]) 
def entropy_of_list(a_list): 
    total_instances = len(a_list) 
    class_counts = a_list.value_counts()
    probs = class_counts / total_instances 
    return entropy(probs) 
def information_gain(df, split_attribute_name, target_attribute_name): 
    total_entropy = entropy_of_list(df[target_attribute_name]) 
    subset_entropy = df.groupby(split_attribute_name)[target_attribute_name].apply(entropy_of_list) 
    subset_sizes = df.groupby(split_attribute_name).size() 
    weighted_entropy = (subset_entropy * subset_sizes / len(df)).sum() 
    return total_entropy - weighted_entropy 
# Read the dataset 
df = pd.read_csv('/content/id3.csv') 
# Get attribute names and remove the target attribute 
attribute_names = list(df.columns) 
target_attribute_name = 'Answer' 
attribute_names.remove(target_attribute_name) 
# Build the decision tree 
tree = id3(df, target_attribute_name, attribute_names) 
# Print the resultant decision tree 
print("Decision Tree:") 
print(tree)