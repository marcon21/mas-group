import numpy as np
from itertools import permutations
import random

def generate_voting_matrices(num_preferences, num_voters, num_examples):
    # Generate all possible letters (options) based on the number of preferences
    options = [chr(65 + i) for i in range(num_preferences)]  # ASCII 'A' is 65

    # Create an empty list to store the generated matrices
    matrices = []

    # Generate all possible permutations of the options
    all_permutations = list(permutations(options))

    # Generate the required number of examples
    for _ in range(num_examples):
        # Randomly select 'num_voters' permutations for the matrix
        selected_permutations = random.sample(all_permutations, num_voters)

        # Convert the permutations into a numpy array (matrix) and append to the list
        matrix = np.array(selected_permutations).T  # Transpose to get preferences as columns
        matrices.append(matrix)

    return matrices

# Function to interact with the user and generate matrices
def main():
    x = int(input("Enter the number of preferences (p): "))
    y = int(input("Enter the number of voters (v): "))
    n = int(input("Enter the number of examples (n): "))

    # Ensure valid input
    if x > 26:
        print("The number of preferences cannot exceed 26 (letters in the English alphabet).")
        return
    
    matrices = generate_voting_matrices(x, y, n)

    # Display the generated matrices
    for i, matrix in enumerate(matrices, 1):
        print(f"\nMatrix #{i}:")
        for row in matrix:
            print(" ", " ".join(row))

# Run the main function if the script is executed
if __name__ == "__main__":
    main()