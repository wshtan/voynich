""" decipherer.py - A minimum implementation of the algorithm.

Slow and dumb but hopefully easy to understand.
"""

import random
import time

SHOULD_PRINT_DEBUG = True  # Change this to False to remove debug messages.
CORPUS_FILE_PATH = "./data/warpeace_input.txt"
SCRAMBLED_TEXT = """v XwaNg cPSRSP o iwPS occNKkoMKwj HoDSg
cPwtSkM, (MxS kwgKjf wj xX1 XoD
HKM xoPg, v XoD ajoHNS Mw DwNpS KM iyDSNR HaM SpSjMaoNNy oRMSP 
XwPUKjf XKMx
iy MSoiioMSD XS DwNpSg KM MwfSMxSP).
"""


def debug(*args):
    if SHOULD_PRINT_DEBUG:
        print(*args)


class LanguageModel:
    """The bi-gram (Markov) language model. """

    def __init__(self, corpus):
        """
        Example:
            CORPUS_FILE_PATH = "./data/warpeace_input.txt"
            war_and_peace_text = str(open(CORPUS_FILE_PATH).read())
            english_model = LanguageModel(corpus=war_and_peace_text)
        """
        self.alphabet = list(set(corpus))
        self.index_of_symbol = {
            self.alphabet[i]: i for i in range(len(self.alphabet))
        }
        # Create an n-by-n matrix, where n = len(alphabets):
        self.alphabet_matrix = [
            [0] * len(self.alphabet) for _ in range(len(self.alphabet))
        ]
        for i in range(len(corpus) - 1):
            this_symbol_index = self.index_of_symbol[corpus[i]]
            next_symbol_index = self.index_of_symbol[corpus[i + 1]]
            self.alphabet_matrix[this_symbol_index][next_symbol_index] += 1

    def compute_score_of(self, text):
        """
        Example:
            english_model.compute_score_of("This is a sunny day!"))  # larger
            english_model.compute_score_of("this si a bunny ady!"))  # smaller
        """
        score = 1
        for i in range(len(text) - 1):
            this_symbol_index = self.index_of_symbol[text[i]]
            next_symbol_index = self.index_of_symbol[text[i + 1]]
            likelihood = self.alphabet_matrix[this_symbol_index][next_symbol_index]
            if likelihood != 0:
                score *= likelihood
        return score


def apply(permutation_map, text):
    return ("").join(
        [permutation_map[symbol] if symbol in permutation_map else symbol for symbol in text]
    )


def decipher(language_model, scrambled_text, iterations):
    # Run Metropolis-Hasting algorithm:
    current_map = {symbol: symbol for symbol in language_model.alphabet}
    current_score = language_model.compute_score_of(scrambled_text)
    debug("metro: init: current score: ", current_score)
    accepted_count = 0
    for iteration_count in range(iterations):
        # propose a move:
        k1, k2 = random.sample(current_map.keys(), 2)
        maybe_next_map = dict(current_map)
        maybe_next_map[k1] = current_map[k2]
        maybe_next_map[k2] = current_map[k1]
        maybe_next_text = apply(maybe_next_map, scrambled_text)
        next_score = language_model.compute_score_of(maybe_next_text)
        should_accept = (
            True
            if next_score > current_score
            else random.random() <= (next_score / current_score)
        )
        if should_accept:
            debug("metro: accept!")
            debug(f"metro: acc / iter: [{accepted_count} / {iteration_count}]")
            debug(f"score: {next_score}")
            debug("metro: current text: ", maybe_next_text)
            current_score = next_score
            current_map = maybe_next_map
            accepted_count += 1


if __name__ == "__main__":
    learning_started_at = time.time()
    with open(CORPUS_FILE_PATH) as corpus_file:
        english_corpus = str(corpus_file.read())
        english_model = LanguageModel(corpus=english_corpus)
        debug(
            "main(): this is a sunny day!",
            english_model.compute_score_of("this is a sunny day!"),
        )  # larger
        debug(
            "main(): htis si a bunny ady!",
            english_model.compute_score_of("htis si a bunny ady!"),
        )  # smaller
    learning_ended_at = time.time()
    learning_time_used = learning_ended_at - learning_started_at
    initial_score = english_model.compute_score_of(SCRAMBLED_TEXT)
    debug("main(): scrambled_text: ", SCRAMBLED_TEXT)
    print("main(): Press enter to start...", end="")
    input()
    # Run Metropolis-Hasting algorithm:
    decipher_started_at = time.time()
    decipher(english_model, SCRAMBLED_TEXT, iterations=50000)
    decipher_ended_at = time.time()
    decipher_time_used = decipher_ended_at - decipher_started_at
    debug("main(): Benchmark:")
    debug("main():     - leanring_time_used (s): ", learning_time_used)
    debug("main():     - initial_score: ", initial_score)
    debug("main():     - decipher_time_used: ", decipher_time_used)
    debug(
        "main():     - decipher_time_used per iter (ms): ",
        decipher_time_used * 1000 / 50000,
    )
