import os
import argparse
import csv
from datasets import load_from_disk
def to_literal_string(text):
    # return text.encode("unicode_escape").decode("utf-8")
    return text

def main():
    parser = argparse.ArgumentParser(description="Process a list of strings and save to a CSV file.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory.")
    args = parser.parse_args()

    # Sample structure
    output_file = os.path.join(args.dataset_path, "torchtune_data.csv")

    print("Processing data...")
    ds = load_from_disk(args.dataset_path)
    texts = ds['text']

    # Prepare data for CSV
    rows = []
    for text in texts:
        text = to_literal_string(text)
        if "Answer:" in text:
            input_part, output_part = text.split("Answer:", 1)
            output_part = "Answer:" + output_part  # Add back the "Answer:" prefix
            rows.append({"input": input_part.strip(), "output": output_part.strip()})
        else:
            print(f"Skipping string as it lacks 'Answer:': {text}")

    # Write to CSV
    print(f"Saving processed data to {output_file}...")
    os.makedirs(args.dataset_path, exist_ok=True)  # Ensure directory exists
    with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["input", "output"])
        writer.writeheader()
        writer.writerows(rows)

    print("Data saved successfully.")

if __name__ == "__main__":
    main()
