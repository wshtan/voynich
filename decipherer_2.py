import math
import random
import time

SHOULD_PRINT_DEBUG = True
CORPUS_FILE_PATH = "./data/warpeace_input.txt"
SCRAMBLED_TEXT_FILE_PATH = "./data/shakespeare_scrambled.txt"
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def debug(*args):
    if SHOULD_PRINT_DEBUG:
        print(*args)


class LanguageModel:
    def __init__(self, corpus):
        """
        Example:
            CORPUS_FILE_PATH = "./data/warpeace_input.txt"
            war_and_peace_text = str(open(CORPUS_FILE_PATH).read())
            english_model = LanguageModel(corpus=war_and_peace_text)
        """
        self.alphabets = list(set(corpus))
        self.index_of_alphabet = {
            self.alphabets[i]: i for i in range(len(self.alphabets))
        }
        # Create an n-by-n matrix, where n = len(alphabets):
        self.alphabet_matrix = [
            [0] * len(self.alphabets) for _ in range(len(self.alphabets))
        ]
        for i in range(len(corpus) - 1):
            this_alphabet_index = self.index_of_alphabet[corpus[i]]
            next_alphabet_index = self.index_of_alphabet[corpus[i + 1]]
            self.alphabet_matrix[this_alphabet_index][next_alphabet_index] += 1

    def compute_log_score_of(self, text):
        """
        Example:
            english_model.compute_log_score_of("This is a sunny day!"))  # larger
            english_model.compute_log_score_of("this si a bunny ady!"))  # smaller
        """
        score = 0
        for i in range(len(text) - 1):
            this_alphabet_index = self.index_of_alphabet[text[i]]
            next_alphabet_index = self.index_of_alphabet[text[i + 1]]
            likelihood = self.alphabet_matrix[this_alphabet_index][next_alphabet_index]
            if likelihood != 0:
                score += math.log(likelihood)
            else:
                score -= 12
        return score


def apply(permutation_map, text):
    return ("").join(
        [permutation_map[char] if char in permutation_map else char for char in text]
    )


def scramble(plain_text):
    map_from = list(LETTERS)
    map_to = list(LETTERS)
    random.shuffle(map_to)
    scramble_map = {map_from[i]: map_to[i] for i in range(len(LETTERS))}
    return apply(scramble_map, plain_text)


def decipher(language_model, scrambled_text, iterations):
    # Run Metropolis-Hasting algorithm:
    current_map = {alphabet: alphabet for alphabet in LETTERS}
    current_log_score = language_model.compute_log_score_of(scrambled_text)
    debug("metro: init: current score: ", current_log_score)
    iteration_count = 0
    accepted_count = 0
    for _ in range(iterations):
        # propose a move:
        #k1, k2, k3, k4 = random.sample(current_map.keys(), 4)
        k1, k2 = random.sample(current_map.keys(), 2)
        maybe_next_map = dict(current_map)
        maybe_next_map[k1] = current_map[k2]
        maybe_next_map[k2] = current_map[k1]
        #maybe_next_map[k3] = current_map[k4]
        #maybe_next_map[k4] = current_map[k3]
        # NOTE (wtan 2023-03-24) Doing multiple swaps actually works in here!
        # NOTE (wtan 2023-03-24) Need evidence
        maybe_next_text = apply(maybe_next_map, scrambled_text)
        next_log_score = language_model.compute_log_score_of(maybe_next_text)
        should_accept = (
            True
            if next_log_score > current_log_score
            else math.log(random.random()) <= (next_log_score - current_log_score)
        )
        if should_accept:
            debug("metro: accept!")
            debug(f"metro: acc / iter: [{accepted_count} / {iteration_count}]")
            debug(f"score: {next_log_score}")
            debug("metro: current text: ", maybe_next_text[:128])
            current_log_score = next_log_score
            current_map = maybe_next_map
            accepted_count += 1
        iteration_count += 1


if __name__ == "__main__":
    learning_started_at = time.time()
    with open(CORPUS_FILE_PATH) as corpus_file:
        english_corpus = str(corpus_file.read())
        debug("main(): reading corpus...")
        english_model = LanguageModel(corpus=english_corpus)
        debug("main(): reading corpus... Done.")
        debug(
            "main(): this is a sunny day!",
            english_model.compute_log_score_of("this is a sunny day!"),
        )  # larger
        debug(
            "main(): htis si a bunny ady!",
            english_model.compute_log_score_of("htis si a bunny ady!"),
        )  # smaller
    learning_ended_at = time.time()
    learning_time_used = learning_ended_at - learning_started_at
    # scrambled_text = scramble(ORIGINAL_TEXT)
    scrambled_text = str(open(SCRAMBLED_TEXT_FILE_PATH).read())
    initial_log_score = english_model.compute_log_score_of(scrambled_text)
    debug("main(): scrambled_text (first 128 chars): ", scrambled_text[:128])
    print("Start?", end="")
    input()
    # Run Metropolis-Hasting algorithm:
    decipher_started_at = time.time()
    decipher(english_model, scrambled_text, iterations=20000)
    decipher_ended_at = time.time()
    decipher_time_used = decipher_ended_at - decipher_started_at
    debug("main(): Benchmarks:")
    debug("main():     - leanring_time_used (s): ", learning_time_used)
    debug("main():     - initial_log_score: ", initial_log_score)
    debug("main():     - decipher_started_at: ", decipher_started_at)
    debug("main():     - decipher_ended_at: ", decipher_ended_at)
    debug("main():     - decipher_time_used: ", decipher_time_used)
    debug(
        "main():     - decipher_time_used per iter (ms): ",
        decipher_time_used * 1000 / 20000,
    )
