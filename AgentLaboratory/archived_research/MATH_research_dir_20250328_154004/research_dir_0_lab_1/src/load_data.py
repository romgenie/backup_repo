from datasets import load_dataset

# Load the MATH500 test dataset using HuggingFace
MATH_test_set = load_dataset("HuggingFaceH4/MATH-500")["test"]

# Prepare the dataset by extracting relevant fields for each example.
prepared_data = []
for example in MATH_test_set:
    # Assume each example contains a "problem" key for the question and "solution" key for the answer.
    problem_text = example.get("problem", "").strip()
    solution_text = example.get("solution", "").strip()
    # Create a simple dictionary with the question and solution for further processing.
    prepared_data.append({
        "question": problem_text,
        "solution": solution_text
    })

# Show a few examples to verify the prepared data
print("Sample prepared data:")
for entry in prepared_data[:3]:
    print(entry)