import pandas as pd
import re
from typing import List, Tuple, Dict
from rmh_extractor import RmhExtractor, RmhWord
from tqdm import tqdm

class BIOBiasTagger:
    def __init__(self, keyword_files: Dict[str, str]):
        self.keyword_dict = self._load_keywords(keyword_files)
        # Flatten all keywords into a list of (category, keyword) tuples
        self.all_keywords = []
        for category, keywords in self.keyword_dict.items():
            for kw in keywords:
                self.all_keywords.append((category, kw))
        # Sort by number of tokens in keyword, descending (longest phrases first)
        self.all_keywords.sort(key=lambda x: -len(x[1].split()))

    def _load_keywords(self, keyword_files: Dict[str, str]) -> Dict[str, List[str]]:
        keyword_dict = {}
        for category, file_path in keyword_files.items():
            df = pd.read_csv(file_path, sep="\t", skiprows=1, header=None)
            keyword_dict[category] = [str(word).strip().lower() for word in df[0] if pd.notna(word)]
        return keyword_dict

    def label_sentence(self, tokens: List[str]) -> List[str]:
        labels = ["O"] * len(tokens)
        lowered_tokens = [t.lower() for t in tokens]

        for category, keyword in self.all_keywords:
            keyword_tokens = keyword.split()
            kw_len = len(keyword_tokens)
            for i in range(len(tokens) - kw_len + 1):
                # Only match if all tokens in this span are still "O"
                if all(l == "O" for l in labels[i:i+kw_len]):
                    if lowered_tokens[i:i+kw_len] == keyword_tokens:
                        labels[i] = f"B-{category}"
                        for j in range(1, kw_len):
                            labels[i + j] = f"I-{category}"
        return labels

    def label_dataset(self, dataset: List[List[str]]) -> List[List[str]]:
        """
        Args:
            dataset: List of tokenized sentences, where each sentence is a list of tokens.
        Returns:
            A list of label lists corresponding to each sentence.
        """
        return [self.label_sentence(sentence) for sentence in dataset]
    

keyword_files = {
    "ADDICTION" : "../bias_vocab/addiction.tsv",
    "DISABILITY" : "../bias_vocab/disability.tsv",
    "ORIGIN" : "../bias_vocab/ethnicity_nationality.tsv",
    "GENERAL" : "../bias_vocab/general.tsv",
    "LGBTQIA" : "../bias_vocab/LGBTQIA+.tsv",
    "LOOKS" : "../bias_vocab/looks.tsv",
    "PERSONAL" : "../bias_vocab/personality_traits.tsv",
    "PROFANITY" : "../bias_vocab/profanity.tsv",
    "RELIGION" : "../bias_vocab/religion.tsv",
    "SEXUAL" : "../bias_vocab/sexual.tsv",
    "SOCIAL_STATUS" : "../bias_vocab/social_status.tsv",
    "STUPIDITY" : "../bias_vocab/stupidity.tsv",
    "VULGAR" : "../bias_vocab/vulgar.tsv",
    "WOMEN" : "../bias_vocab/women.tsv",
    
}
for i in ["bland_2002", "bland_2003", "bland_2004", "bland_2005", "heimur", "jonas", "silfuregils", "sunnlenska", "viljinn" ]:
    # Initialize extractor and gather sentences
    RMH = RmhExtractor(folder="../../RMH/IGC-Social-22.10.ana/Forums/bland/test/"+i)
    sents = []
    sents_lemmatized = []
    current_sent = []
    current_sent_lemmatized = []
    for word in RMH.extract(forms=True, lemmas=True, pos=True):
        if word.word_form == ".":
            current_sent.append(word.word_form)
            sents.append(current_sent)
            current_sent_lemmatized.append(word.lemma)
            sents_lemmatized.append(current_sent_lemmatized)
            current_sent = []
            current_sent_lemmatized = []
        else:
            current_sent.append(word.word_form)
            current_sent_lemmatized.append(word.lemma)
    # Catch any final sentence without a trailing period
    if current_sent:
        sents.append(current_sent)
        sents_lemmatized.append(current_sent_lemmatized)

    # Tag the sentences based on lemmas, with progress bar
    tagger = BIOBiasTagger(keyword_files)
    labels = []
    for sentence in tqdm(sents_lemmatized, desc="Labelling sentences"):
        labels.append(tagger.label_sentence(sentence))

    # Write output to a file, formatted as: word_form<TAB>BIO-tag, sentences separated by an empty line
    output_path = "../biolabelled/"+i+".txt"
    with open(output_path, "w+", encoding="utf-8") as f:
        for word_forms, tag_seq in zip(sents, labels):
            for wf, tag in zip(word_forms, tag_seq):
                f.write(f"{wf}\t{tag}\n")
            f.write("\n")

    print(f"BIO-tagged output written to {output_path}")