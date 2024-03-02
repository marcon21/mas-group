import numpy as np
from itertools import permutations
import random
import json

def generate_voting_matrices(num_preferences, num_voters, num_examples):
    # Generate all possible letters (options) based on the number of preferences
    options = [chr(65 + i) for i in range(num_preferences)]  # ASCII 'A' is 65

    # Create an empty list to store the generated matrices
    matrices = []

    # Generate the required number of examples
    for _ in range(num_examples):
        matrix = []
        for __ in range(num_voters):
            # Randomly select 'num_voters' permutations for the matrix, allowing for duplicates
            shuffled_options= random.sample(options, len(options))
            matrix.append(shuffled_options)

        # Convert the permutations into a numpy array (matrix) and append to the list
        matrix = np.array(matrix).T.tolist()  # Transpose to get preferences as columns
        matrices.append(matrix)

    return matrices

# Function to interact with the user and generate matrices
def main():
    num_preferences = int(input("Enter the number of preferences (p): "))
    num_voters = int(input("Enter the number of voters (v): "))
    num_examples = int(input("Enter the number of examples (n): "))

    # Ensure valid input
    if num_preferences > 26:
        print("The number of preferences cannot exceed 26 (letters in the English alphabet).")
        return

    matrices = generate_voting_matrices(num_preferences, num_voters, num_examples)

    # Prepare data for JSON
    data = {f'voting{i + 1}': matrix for i, matrix in enumerate(matrices)}

   # Write to JSON file
    with open('input/voting_result.json', 'w') as json_file:
        # Manually construct the JSON string for matrices to control formatting
        json_strings = [f'"{key}": ' + '\n[\n' + ',\n'.join('[' + ', '.join(f'"{item}"' for item in row) + ']' for row in matrix) + '\n]' for key, matrix in data.items()]
        combined_json_string = '{\n' + ', \n'.join(json_strings) + '\n}'
        json_file.write(combined_json_string)


    print(f"Generated {num_examples} matrices with {num_preferences} preferences and {num_voters} voters. Results saved to 'voting_result.json'.")
    return num_examples, num_preferences

# Run the main function if the script is executed
if __name__ == "__main__":
    main()