import random
import math
import time
import heapq
import copy
import numpy as np
import pandas as pd

small_dataset_original = np.loadtxt('/workspaces/CS170-Project-2-Abhiram-Mupparaju/small-test-dataset-2-2.txt')
large_dataset_original = np.loadtxt('/workspaces/CS170-Project-2-Abhiram-Mupparaju/large-test-dataset-2.txt')
titanic_dataset_original = np.loadtxt('/workspaces/CS170-Project-2-Abhiram-Mupparaju/titanic-clean-2.txt')
# lines 3 - 7 copied using templates from my assignments in CS171

#- Group: Abhiram Mupparaju - amupp001 - Lecture Session 1 - Discussion section 021
# DatasetID: 
# Small Dataset Results:
#   Forward: Feature Subset: {3, 5}, Acc: 0.92 (92%)
#   Backward: Feature Subset: {2, 3, 4, 5} Acc: 0.83 (83.0%)
# Large Dataset Results:
#   Forward: Feature Subset: {1, 27}, Acc: 0.955 (95.5%)
#   Backward: Feature Subset: {1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40}, Acc: 0.711 (71.1%)


def min_max_normalize(dataset): #made with assistance from geeksforgeeks cited in report
    # Separate first column of labels from features
    labels = dataset[:, 0]
    features = dataset[:, 1:]

    # Calculate min and max for each feature column
    min_vals = features.min(axis=0)
    max_vals = features.max(axis=0)

    # Calculate range for each feature column
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1 # Avoid division by zero

    # Apply Min-Max scaling
    normalized_features = (features - min_vals) / range_vals

    # rebuild dataset with labels
    normalized_dataset = np.hstack((labels[:, np.newaxis], normalized_features))
    return normalized_dataset

# Run normalization on all datasets
small_dataset = min_max_normalize(small_dataset_original)
large_dataset = min_max_normalize(large_dataset_original)
titanic_dataset = min_max_normalize(titanic_dataset_original)

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
                # Adjust for 0-based indexing if features are 1-based
                feature_vector = [instance[feature_index] for feature_index in selected_features]
            else: #use all features if not specified
                feature_vector = instance[1:]

            processed_data.append((feature_vector, label))

        self.training_instance = processed_data

    def test(self, test_instance):
        if self.training_instance is None or not self.training_instance: #check if lcassifier has been trained
            raise ValueError("Error: Classifier has not been trained")

        min_distance = float('inf') #initialize with infinty
        predicted_class_label = None

        for training_feature_vector, training_label in self.training_instance:
            check_distance = self.euclidean_distance(test_instance, training_feature_vector)

            if check_distance < min_distance:
                min_distance = check_distance
                predicted_class_label = training_label

        return predicted_class_label # Only return predicted label, time calculation handled by caller


class Leave_One_Out_Validator: #input feature subset, NN classifier and the dataset
    def __init__(self):
        pass

    def evaluate(self, nnclassifier, feature_subset, dataset):
        correct_class = 0
        total_instances = len(dataset)
        total_nn_classifier_time = 0.0  # Initialize total time for NN classifier operations (train + test)

        if total_instances == 0: #safety check
            print("Error: Dataset is empty")
            return 0.0, 0.0 # Return accuracy, total_nn_classifier_time

        if not feature_subset: # Handle empty feature_subset for 'no features' scenario
            majority_class_correct = 0
            for i in range(total_instances):
                training_set = [dataset[j] for j in range(total_instances) if j != i]
                test_instance_curr = dataset[i]
                correct_label = test_instance_curr[0]

                # Determine majority class in the training set
                class_counts = {}
                for instance in training_set:
                    label = instance[0]
                    class_counts[label] = class_counts.get(label, 0) + 1

                if not class_counts: #safety case for if datyaset has only 1 value
                    predicted_majority_class = 0
                else:
                    # Find the class with the maximum count
                    max_count = 0
                    predicted_majority_class = 0
                    for class_label, count in class_counts.items():
                        if count > max_count:
                            max_count = count
                            predicted_majority_class = class_label
                        elif count == max_count: # Tie-breaking: smaller label wins or simply pick first seen
                            if predicted_majority_class == 0 or class_label < predicted_majority_class:
                                predicted_majority_class = class_label # In case of a tie pick smallest label to be consistent

                if predicted_majority_class == correct_label:
                    majority_class_correct += 1
            accuracy = (majority_class_correct / total_instances) * 100
            return accuracy, 0.0 # NN classifier time is 0 for this path

        else:
            for i in range(total_instances):
                training_set = [dataset[j] for j in range(total_instances) if j != i] #leave one instance out

                test_instance_curr = dataset[i] #prepare current instance for test
                correct_label = test_instance_curr[0]

                #get features for test
                test_instance_features = [test_instance_curr[feature_index] for feature_index in feature_subset]

                #get time spent on nn_classifer step
                nn_start_time = time.time()
                nnclassifier.train(training_set, feature_subset)
                predicted_label = nnclassifier.test(test_instance_features)
                nn_end_time = time.time()
                total_nn_classifier_time += (nn_end_time - nn_start_time)

                if predicted_label == correct_label:
                    correct_class += 1

            accuracy = (correct_class / total_instances) * 100
            return accuracy, total_nn_classifier_time # Return accuracy and NN_classifier time


def forward_selection(total_features_list, dataset, validator):
    curr_features = [] 

    dummy_nn_classifier = NN_classifier()

    # No features base case
    initial_accuracy_no_features, _ = validator.evaluate(dummy_nn_classifier, [], dataset)
    max_score = initial_accuracy_no_features # The overall best found so far
    max_feature_set = [] # The overall best feature set found so far (empty initially)

    print(f'Running nearest neighbor with no features (default rate), using "leaving-one-out" evaluation, I\nget an accuracy of {initial_accuracy_no_features:.1f}%')
    print("Beginning search.")

    # Track accuracy of curr_features
    current_level_base_accuracy = initial_accuracy_no_features

    while True:
        feature_to_add_in_current_step = None
        best_accuracy_in_current_step = 0.0 # Tracks highest accuracy achieved by adding 1 feature
        best_set_candidate_for_printing_in_current_step = [] # The feature set that yields best_accuracy_in_current_step

        features_remaining = [f for f in total_features_list if f not in curr_features]

        if not features_remaining: # break if no features left
            print("\nNo more features to add.")
            break 

        print(f"\n    On the current level, adding one feature to the current set: {{{', '.join(map(str, sorted(curr_features)))}}}.")

        for feature_candidate in features_remaining:
            test_set_candidate = sorted(curr_features + [feature_candidate])
            feature_indices = [f_int for f_int in test_set_candidate]

            temp_nn_classifier = NN_classifier()
            score, _ = validator.evaluate(temp_nn_classifier, feature_indices, dataset)
            print(f"        Using feature(s) {{{', '.join(map(str, test_set_candidate))}}} accuracy is {score:.1f}%")

            if score > best_accuracy_in_current_step:
                best_accuracy_in_current_step = score
                feature_to_add_in_current_step = feature_candidate
                best_set_candidate_for_printing_in_current_step = list(test_set_candidate)

        # Print the best accuracy for this iteration
        if feature_to_add_in_current_step is not None:
             print(f"    Feature set {{{', '.join(map(str, sorted(best_set_candidate_for_printing_in_current_step)))}}} was best, accuracy is {best_accuracy_in_current_step:.1f}%")
        else: # safety case in case accuracy doesn't imporve on non empty set
            print("    No feature improved accuracy in this round.")
            break


        # Check if the best accuracy was found by adding a feature in this iteration
        if best_accuracy_in_current_step < current_level_base_accuracy:
            print(f"(Warning, Accuracy has decreased. Stopping search.)")
            break 

        # add feature since accuracy increased
        curr_features.append(feature_to_add_in_current_step)
        curr_features.sort()
        current_level_base_accuracy = best_accuracy_in_current_step # Update the base for the next round


        # 4. Compare accuracy and take best
        if current_level_base_accuracy > max_score:
            max_score = current_level_base_accuracy
            max_feature_set = list(curr_features)
            print(f"Feature set {{{', '.join(map(str, max_feature_set))}}} was best, accuracy is {max_score:.1f}%")
        elif current_level_base_accuracy == max_score:
            pass
        else: 
            print(f"(Warning, Accuracy has decreased! Continuing search in case of local maxima)")

    return max_feature_set, max_score

def backward_elimination(total_features_list, dataset, validator):
    curr_features = sorted([int(f) for f in total_features_list]) 

    dummy_nn_classifier = NN_classifier()
    # Initial evaluation with all features, base global max
    max_score, _ = validator.evaluate(dummy_nn_classifier, curr_features, dataset)
    max_feature_set = list(curr_features)

    print(f"Starting Backward Elimination with all features: {{{', '.join(map(str, curr_features))}}}. Initial score: {max_score:.1f}%")
    print("Beginning search.")

    while True:
        feature_to_remove_in_current_step = None # The feature to remove that will get the highest accuracy
        best_accuracy_in_current_step = 0.0 # Accuracy achieved by removing said feature

        next_curr_features_candidate = []

        if len(curr_features) <= 1:
            break

        print(f"\n    On the current level, removing one feature from the current set: {{{', '.join(map(str, sorted(curr_features)))}}}.")

        for feature_candidate_to_remove in curr_features:
            test_set_candidate = sorted([f for f in curr_features if f != feature_candidate_to_remove]) # New set if this feature is removed

            temp_nn_classifier = NN_classifier()
            score, _ = validator.evaluate(temp_nn_classifier, test_set_candidate, dataset)
            print(f"        Using feature(s) {{{', '.join(map(str, test_set_candidate))}}} accuracy is {score:.1f}%") 

            if score > best_accuracy_in_current_step:
                best_accuracy_in_current_step = score
                feature_to_remove_in_current_step = feature_candidate_to_remove
                next_curr_features_candidate = list(test_set_candidate) 

        # Compare the current best accuracy with max_score
        if best_accuracy_in_current_step > max_score:
            max_score = best_accuracy_in_current_step
            max_feature_set = list(next_curr_features_candidate)
            curr_features = list(next_curr_features_candidate)
            curr_features.sort()
            print(f"Feature set {{{', '.join(map(str, max_feature_set))}}} was best, accuracy is {max_score:.1f}%")
        else:
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)") 
            if feature_to_remove_in_current_step is not None:
                print(f"Feature set {{{', '.join(map(str, next_curr_features_candidate))}}} was best, accuracy is {best_accuracy_in_current_step:.1f}%")
            break 

    return max_feature_set, max_score

def main(): #main function to run the program
    print(f"Welcome to My Feature Selection Algorithm.") # Changed name as per request

    dataset_filename = input("Type in the name of the file to test (small, large, titanic): ")
    print("Please wait while I normalize the data... ", end='')

    selected_dataset = None
    try:
        if dataset_filename == "small":
            selected_dataset = small_dataset # Use normalized small_dataset
        elif dataset_filename == "large":
            selected_dataset = large_dataset # Use normalized large_dataset
        elif dataset_filename == "titanic":
            selected_dataset = titanic_dataset # Use normalized titanic_dataset
        else:
            print("Inavlid file name, restart")

        print("Done!")
    except Exception as e: # safety check, exit if error in normalizing
        print(f"Error loading or normalizing dataset: {e}")
        return

    num_features = selected_dataset.shape[1] - 1
    total_features_list = list(range(1, num_features + 1))

    print(f"This dataset has {num_features} features (not including the class attribute), with {selected_dataset.shape[0]} instances.")
    print("Type the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    print("3) Special Algorithm.")

    algorithm_choice = input()

    validator = Leave_One_Out_Validator()

    if algorithm_choice == "1":
        print("1) Forward Selection")
        best_feature_set, best_score = forward_selection(total_features_list, selected_dataset, validator)
        print(f"Finished search!! The best feature subset is {{{', '.join(map(str, sorted(best_feature_set)))}}}, which has an accuracy of {best_score:.1f}%")
    elif algorithm_choice == "2":
        print("2) Backward Elimination")
        best_feature_set, best_score = backward_elimination(total_features_list, selected_dataset, validator)
        print(f"Finished search!! The best feature subset is {{{', '.join(map(str, sorted(best_feature_set)))}}}, which has an accuracy of {best_score:.1f}%")
    elif algorithm_choice == "3":
        print("3) Special Algorithm.")
    else:
        print("Invalid")

if __name__ == "__main__":
   main()