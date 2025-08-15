"""
This script samples sentences from automatically annotated CoNLL-style files for NER bias analysis.

- Scans all .txt files in the input directory.
- For each NER category, samples up to 5 sentences where a B-tag is directly followed by an I-tag, and 5 where it is not.
- Ensures sampled words are not too similar (by edit distance).
- Fills up to 100 sentences with random samples if needed.
- Writes the sampled sentences to an output file.

Input: Directory of .txt files in CoNLL format (word<TAB>tag per line, blank line between sentences).
Output: A file with sampled sentences, separated by blank lines.
"""

import os
import random
from collections import defaultdict

input_dir = "manual_test"
output_file = "sampled_sentences.txt"
categories = [
    "B-ADDICTION", "I-ADDICTION", "B-DISABILITY", "I-DISABILITY", "B-ORIGIN", "I-ORIGIN", "B-GENERAL", "I-GENERAL",
    "B-LGBTQIA", "I-LGBTQIA", "B-LOOKS", "I-LOOKS", "B-PERSONAL", "I-PERSONAL", "B-PROFANITY", "I-PROFANITY",
    "B-RELIGION", "I-RELIGION", "B-SEXUAL", "I-SEXUAL", "B-SOCIAL_STATUS", "I-SOCIAL_STATUS",
    "B-STUPIDITY", "I-STUPIDITY", "B-VULGAR", "I-VULGAR", "B-WOMEN", "I-WOMEN"]

def word_distance(w1, w2):
    """
    Compute a simple edit distance between two words.
    Returns 0 if identical, or the edit distance otherwise.
    """

    if w1 == w2:
        return 0
    if abs(len(w1) - len(w2)) > 3:
        return abs(len(w1) - len(w2))
    dp = [[i + j if i * j == 0 else 0 for j in range(len(w2) + 1)] for i in range(len(w1) + 1)]
    for i in range(1, len(w1) + 1):
        for j in range(1, len(w2) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (w1[i - 1] != w2[j - 1])
            )
    return dp[-1][-1]

def get_b_words(sent, b_tag):
    """
    Extract all words in a sentence with the given B-tag.
    """
    words = []
    for line in sent:
        if '\t' in line:
            word, tag = line.split('\t')
            if tag == b_tag:
                words.append(word)
    return words

def is_far_enough(word, chosen_words):
    """
    Check if a word is sufficiently different from all words in chosen_words.
    """
    for w in chosen_words:
        if word_distance(word, w) < 3:
            return False
    return True

# Collect sentences per category
category_to_sentences = defaultdict(list)
all_sentences = []

def read_conll_sentences(file_path):
    """
    Read sentences from a CoNLL-style file.
    Returns a list of sentences, each a list of lines.
    """
    sentences = []
    with open(file_path, encoding="utf-8") as f:
        sentence = []
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                sentence.append(line)
        if sentence:
            sentences.append(sentence)
    return sentences

# Scan all files and collect sentences
for fname in os.listdir(input_dir):
    if fname.endswith(".txt"):
        sentences = read_conll_sentences(os.path.join(input_dir, fname))
        for sent in sentences:
            tags = [line.split('\t')[1] for line in sent if '\t' in line]
            for cat in categories:
                if cat in tags:
                    category_to_sentences[cat].append(sent)
            all_sentences.append(sent)

def b_followed_by_i(sent, b_tag, i_tag):
    """
    Return True if any B-tag in the sentence is directly followed by the corresponding I-tag.
    """
    for idx, line in enumerate(sent[:-1]):
        if '\t' in line:
            word, tag = line.split('\t')
            if tag == b_tag:
                next_line = sent[idx + 1]
                if '\t' in next_line:
                    _, next_tag = next_line.split('\t')
                    if next_tag == i_tag:
                        return True
    return False

def b_not_followed_by_i(sent, b_tag, i_tag):
    """
    Return True if any B-tag in the sentence is NOT directly followed by the corresponding I-tag.
    """
    for idx, line in enumerate(sent):
        if '\t' in line:
            word, tag = line.split('\t')
            if tag == b_tag:
                # If last token, can't be followed by I
                if idx == len(sent) - 1:
                    return True
                next_line = sent[idx + 1]
                if '\t' in next_line:
                    _, next_tag = next_line.split('\t')
                    if next_tag != i_tag:
                        return True
    return False

selected_sentences = set()

for cat in categories:
    if cat.startswith("B-"):
        i_tag = "I-" + cat[2:]
        sents = category_to_sentences[cat]
        random.shuffle(sents)

        # 1. B-class directly followed by I-class
        chosen_words_followed = set()
        sampled_followed = []
        for sent in sents:
            if b_followed_by_i(sent, cat, i_tag):
                b_words = get_b_words(sent, cat)
                for word in b_words:
                    if is_far_enough(word, chosen_words_followed):
                        sampled_followed.append(sent)
                        chosen_words_followed.add(word)
                        break
            if len(sampled_followed) >= 5:
                break
        for sent in sampled_followed:
            selected_sentences.add(tuple(sent))
        if len(sampled_followed) < 5:
            print(f"Warning: Less than 5 sentences found for {cat} with B followed by I. Found {len(sampled_followed)}.")

        # 2. B-class NOT directly followed by I-class
        chosen_words_not_followed = set()
        sampled_not_followed = []
        for sent in sents:
            if b_not_followed_by_i(sent, cat, i_tag):
                b_words = get_b_words(sent, cat)
                for word in b_words:
                    if is_far_enough(word, chosen_words_not_followed):
                        sampled_not_followed.append(sent)
                        chosen_words_not_followed.add(word)
                        break
            if len(sampled_not_followed) >= 5:
                break
        for sent in sampled_not_followed:
            selected_sentences.add(tuple(sent))
        if len(sampled_not_followed) < 5:
            print(f"Warning: Less than 5 sentences found for {cat} with B not followed by I. Found {len(sampled_not_followed)}.")


# If less than 100, fill up with random sentences
if len(selected_sentences) < 100:
    needed = 100 - len(selected_sentences)
    remaining = [tuple(s) for s in all_sentences if tuple(s) not in selected_sentences]
    selected_sentences.update(random.sample(remaining, min(needed, len(remaining))))

# Write to output file
with open(output_file, "w", encoding="utf-8") as out:
    for sent in selected_sentences:
        for line in sent:
            out.write(line + "\n")
        out.write("\n")

print(f"Sampled {len(selected_sentences)} sentences to {output_file}")