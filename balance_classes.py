import random

TARGET_TAGS = {"B-VULGAR", "B-PROFANITY"}
INPUT_FILE = "data/all_balanced.txt"
OUTPUT_FILE = "data/all_balanced2.txt"
SEED = 42  # For reproducibility

def read_sentences(filename):
    with open(filename, "r", encoding="utf-8") as f:
        sentences = []
        current = []
        for line in f:
            if line.strip() == "":
                if current:
                    sentences.append(current)
                    current = []
            else:
                current.append(line.rstrip("\n"))
        if current:
            sentences.append(current)
    return sentences

def contains_target_tag(sentence):
    for line in sentence:
        if line.split()[-1] in TARGET_TAGS:
            return True
    return False

def main():
    random.seed(SEED)
    sentences = read_sentences(INPUT_FILE)
    target_sentences = [s for s in sentences if contains_target_tag(s)]
    other_sentences = [s for s in sentences if not contains_target_tag(s)]

    num_to_remove = len(target_sentences) // 3
    to_remove = set(random.sample(range(len(target_sentences)), num_to_remove))

    balanced_sentences = [
        s for i, s in enumerate(target_sentences) if i not in to_remove
    ] + other_sentences

    # Optionally, shuffle the output
    random.shuffle(balanced_sentences)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sentence in balanced_sentences:
            for line in sentence:
                f.write(line + "\n")
            f.write("\n")

if __name__ == "__main__":
    main()