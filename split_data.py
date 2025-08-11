import random

# Load the full dataset
with open("data/all_balanced.txt", encoding="utf-8") as f:
    text = f.read()

# Split into sentences (each separated by a blank line)
sentences = text.strip().split("\n\n")

# Shuffle for randomness
random.seed(42)
random.shuffle(sentences)

# Compute split sizes
total = len(sentences)
train_split = int(0.8 * total)
dev_split = int(0.1 * total)

# Split the data
train_sents = sentences[:train_split]
dev_sents = sentences[train_split:train_split + dev_split]
test_sents = sentences[train_split + dev_split:]

# Save to new files
with open("train.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(train_sents) + "\n")

with open("dev.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(dev_sents) + "\n")

with open("test.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(test_sents) + "\n")

print(f"Split complete: {len(train_sents)} train, {len(dev_sents)} dev, {len(test_sents)} test")