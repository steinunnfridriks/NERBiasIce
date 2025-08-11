def collect_non_O_sentences(input_path, output_path):
    with open(input_path, encoding="utf-8") as f:
        lines = f.readlines()

    sentences = []
    current = []
    for line in lines:
        line = line.strip()
        if not line:
            if current:
                sentences.append(current)
                current = []
        else:
            current.append(line)
    if current:
        sentences.append(current)

    with open(output_path, "w", encoding="utf-8") as out:
        for sentence in sentences:
            # Only check lines with a tab
            if any(
                (parts := token.split('\t')) and len(parts) == 2 and parts[1] != "O"
                for token in sentence
            ):
                for token in sentence:
                    out.write(token + "\n")
                out.write("\n")

if __name__ == "__main__":
    collect_non_O_sentences(
        "/home/steinunn/hotterncoldertest/biolabelled/frettabladid_2023.txt",
        "/home/steinunn/hotterncoldertest/usable_code/data/frettabladid_2023.txt"
    )