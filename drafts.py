class LanguageTrait:
    def __init__(self, corpus):
        self.alphabets = []
        self.frequencies = {}
        for alphabet in corpus:
            if alphabet in self.frequencies:
                self.frequencies[alphabet] += 1
            else:
                self.alphabets.append(alphabet)
                self.frequencies[alphabet] = 1
        self.frequency_rank = sorted(
            [(a, self.frequencies[a]) for a in self.frequencies.keys()],
            key=lambda tup: tup[1],
            reverse=True   # more frequent goes first (have lower index)
        )
        debug("LanguageTrait.__init__(): frequency_rank: len: ", len(self.frequency_rank))
        debug("LanguageTrait.__init__(): frequency_rank: ", self.frequency_rank)
        self.index_of_alphabet = {
            self.alphabets[i] : i
            for i in range(len(self.alphabets))
        }
        debug("LanguageTrait.__init__(): index_of_alphabet: ", self.index_of_alphabet)
        self.alphabet_matrix = [[0] * len(self.alphabets) for _ in range(len(self.alphabets))]
        for i in range(len(corpus)-1):
            this_alphabet_index = self.index_of_alphabet[corpus[i]]
            next_alphabet_index = self.index_of_alphabet[corpus[i+1]]
            self.alphabet_matrix[this_alphabet_index][next_alphabet_index] += 1
        debug("LanguageTrait.__init__(): alphabet_matrix: ", self.alphabet_matrix)

    def compute_score(self, data):
        """
        Example:
            english_trait.compute_score("This is a sunny day!"))  # larger
            english_trait.compute_score("this si a bunny ady!"))  # smaller
        """
        ret = 1
        for i in range(len(data)-1):
            this_alphabet_index = self.index_of_alphabet[data[i]]
            next_alphabet_index = self.index_of_alphabet[data[i+1]]
            likelihood = self.alphabet_matrix[this_alphabet_index][next_alphabet_index]
            if likelihood != 0:
                ret *= likelihood
            else:
                ret // 2
                # The score function in the given code is very advance and very efficient.
                # TODO (wtan 2023-03-24) Understand that.
        return ret




def generate_language(language_trait, length):
    def select_index(probabilities):
        u = random.random()
        s = 0
        for i in range(len(probabilities)):
            s += probabilities[i]
            if u < s:
                return i
        return -1  # should never reach here
    # Normalize the alphabet matrix:
    alphabets = language_trait["alphabets"]
    alphabet_matrix = language_trait["alphabet_matrix"]
    normalized_matrix = [[0] * len(alphabets) for _ in range(len(alphabets))]
    for r in range(len(alphabets)):
        s = sum(alphabet_matrix[r])
        if s == 0:
            normalized_matrix[r][r] = 1
        else:
            for c in range(len(alphabets)):
                normalized_matrix[r][c] = alphabet_matrix[r][c] / s
    # Now do a random walk on the normalized matrix:
    result = ['t']
    current_alphabet_index = language_trait["index_of_alphabet"]['t']
    for _ in range(length):
        current_alphabet_index = select_index(normalized_matrix[current_alphabet_index])
        result.append(alphabets[current_alphabet_index])
    return ("").join(result)
