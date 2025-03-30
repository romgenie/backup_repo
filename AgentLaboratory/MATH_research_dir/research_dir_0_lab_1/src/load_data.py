from datasets import load_dataset

# Load the MATH500 test dataset using the HuggingFace dataset
MATH_test_set = load_dataset("HuggingFaceH4/MATH-500")["test"]

# Print the number of test questions available
print("Number of MATH500 test questions:", len(MATH_test_set))

# Print a sample entry to understand the structure
print("Sample test question:", MATH_test_set[0])