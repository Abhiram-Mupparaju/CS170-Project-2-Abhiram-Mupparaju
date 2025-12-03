import random
import math
import heapq
import copy
import numpy as np
import pandas as pd

small_dataset = np.loadtxt('/workspaces/CS170-Project-2-Abhiram-Mupparaju/small-test-dataset-2-2.txt')
large_dataset = np.loadtxt('/workspaces/CS170-Project-2-Abhiram-Mupparaju/large-test-dataset-2.txt')
# lines 3 - 7 copied using templates from my assignments in CS171

def stub_evaluation():

    # Return a random percent between 0.0 and 100.0
    return random.random() * 100.0

class NN_classifier:
    def __init__(self):
        self.training_instance = None

    def euclidean_distance(self, point_1, point_2):
        if len(point_1) != len(point_2): #check if lists have the same dimensionality
            raise ValueError("Error: Lists have different dimensionalities")

        euclidean = [(p1 - p2)**2 for p1, p2 in zip(point_1, point_2)] #calculate the euclidean distance
        return math.sqrt(sum(euclidean))

    def train(self, training_instance, selected_features=None):
        processed_data = []
        for instance in training_instance:
            label = instance[0] #first element is set as label
            if selected_features:
                feature_vector = [instance[feature_index] for feature_index in selected_features]
            else: #use all features if not specified
                feature_vector = instance[1:]

            processed_data.append((feature_vector, label))

        self.training_instance = processed_data


class Leave_One_Out_Validator: #input feature subset, NN classifier and the dataset
    def __init__(self):
        pass

    def evaluate(self, nnclassifier, feature_subset, dataset):
        correct_class = 0
        total_instances = len(dataset)

        if total_instances == 0: #safety check
            print("Error: Dataset is empty")
            return 0.0

        for i in range(total_instances): 
            training_set = [dataset[j] for j in range(total_instances) if j != i] #leave one instance out

            test_instance_curr = dataset[i] #prepare current instance for test
            correct_label = test_instance_curr[0]

            #get features for test
            test_instance_features = [test_instance_curr[feature_index] for feature_index in feature_subset] 

            #call nnclasifier to train with data
            nnclassifier.train(training_set, feature_subset)

            # test classifer with current instance
            predicted_label = nnclassifier.test(test_instance_features)

            if predicted_label == correct_label:
                correct_class += 1

        accuracy = (correct_class / total_instances) * 100
        return accuracy

def forward_selection(total_features):
    curr_features = []
    
    max_score = stub_evaluation() #inialize with stub evalutation
    max_feature_set = [] # Initially empty

    # initial state
    print(f"Using no features and \"random\" evaluation, I get an accuracy of {max_score:.1f}%") #set accuracy to use 1 decimal place
    print("Beginning search.")

    while True:
        feature_to_add = None
        # Max score achievable by adding ONE feature in this iteration
        check_max_score = -1.0

        features_remaining = [f for f in total_features if f not in curr_features]

        if not features_remaining:
            print("No features left to consider. Stopping forward selection.")
            break

        # Find the max feature to add
        for feature in features_remaining:
            test_set = curr_features + [feature]
            score = stub_evaluation()
            print(f"Using feature(s) {{{', '.join(map(str, sorted(test_set)))}}} accuracy is {score:.1f}%")

            if score > check_max_score:
                check_max_score = score
                feature_to_add = feature

        # check compared to exisitng max score
        if check_max_score > max_score:
            curr_features.append(feature_to_add)
            max_score = check_max_score # Update global best score
            max_feature_set = list(curr_features) # Update global best set
            print(f"Feature set {{{', '.join(map(str, sorted(max_feature_set)))}}} was best, accuracy is {max_score:.1f}%")
        else:
            # If the best possible addition in this iteration does not improve the overall best score
            # found so far, then we stop. The trace's "Accuracy has decreased!" implies this is the stopping point.
            print("(Warning, Accuracy has decreased!)") # Simulate the warning from the trace
            # The algorithm has found no further improvement. Stop.
            break

    # Final output is handled by the calling block to match the trace
    return max_feature_set, max_score

def backward_elimination(total_features):
    curr_features = list(total_features) # Start with all features
    max_score = stub_evaluation(curr_features)
    max_feature_set = list(curr_features)

    print(f"Starting Backward Elimination with all features: {curr_features}. Initial score: {max_score:.1f}")

    while True:
        feature_to_remove = None
        check_max_score = -1 # Initialize with a score worse than any possible random score (0-1)

        # If only one feature is left, we can't remove any more
        if len(curr_features) <= 1:
            print("Only one feature left, stopping backward elimination.")
            break

        for feature in curr_features:
            test_set = [f for f in curr_features if f != feature]
            # Evaluate the set *after* removing a feature
            score_after_removal = stub_evaluation(test_set)

            # We are looking for the removal that yields the highest score
            # (or least drop, if the score generally decreases with fewer features)
            if score_after_removal > check_max_score:
                check_max_score = score_after_removal
                feature_to_remove = feature

        # Check if removing the best feature from this iteration improves the overall best score
        if check_max_score > max_score:
            curr_features.remove(feature_to_remove)
            max_score = check_max_score
            max_feature_set = list(curr_features) # Update the best set
            print(f"Removed feature: {feature_to_remove}. Current best score: {max_score:.1f}, Current best set: {max_feature_set}")
        else:
            # No improvement by removing any more features, stop
            print("No further improvement by removing features. Stopping backward elimination.")
            break

    return max_feature_set, max_score

def main(): #main function to run the program
    print(f"Welcome to Feature Selection Algorithm.")

    user_input_features = int(input("Please enter total number of features: "))

    total_features = [] # Initialize an empty list
    for i in range(1, user_input_features + 1):
        total_features.append(str(i))

    print("Type the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    print("3) Special Algorithm.")

    algorithm_choice = input()

    if algorithm_choice == "1":
        print("1) Forward Selection")
        best_feature_set, best_score = forward_selection(total_features)
        print(f"Finished search!! The best feature subset is {{{', '.join(map(str, sorted(best_feature_set)))}}}, which has an accuracy of {best_score:.1f}%")
    elif algorithm_choice == "2":
        print("2) Backward Elimination")
    elif algorithm_choice == "3":
        print("3) Special Algorithm.")
    else:
        print("Invalid")

if __name__ == "__main__":
   main()