from augment_training_data import create_augmented_dataset
from extraction import create_training_example
import json

print("Creating augmented dataset...")
augmented_texts = create_augmented_dataset()
print(f"Generated {len(augmented_texts)} augmented reports")

print("\nGenerating training data...")
training_data = []
for i, text in enumerate(augmented_texts):
    if i % 50 == 0:
        print(f"  Processing {i}/{len(augmented_texts)}...")
    example = create_training_example(text)
    training_data.append(example)

print(f"\nCreated {len(training_data)} training examples")

with open("training_data_augmented.json", "w") as f:
    json.dump(training_data, f, indent=2)

print(f"Saved to training_data_augmented.json")