import random

def stub_evaluation():

    # Return a random percent between 0.0 and 100.0
    return random.random() * 100.0

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