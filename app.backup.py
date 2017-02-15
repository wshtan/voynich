# run_deciphering.py -- Runs on Python3.

import random
import sys
import time
import os
from copy import copy, deepcopy
from optparse import OptionParser

import numpy as np

# import matplotlib.pyplot as plt  # NOTE (wtan 2023-03-18) plt is not used (yet).

def debug(*args):
    print(*args)


## -- utils.py:


def az_list():
    """
    Returns a default a-zA-Z characters list
    """
    cx = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return cx


def generate_random_permutation_map(chars):
    """
    Generate a random permutation map for given character list. Only allowed permutations
    are alphabetical ones. Helpful for debugging

    Arguments:
    chars: list of characters

    Returns:
    p_map: a randomly generated permutation map for each character
    """
    cx = az_list()
    cx2 = az_list()
    random.shuffle(cx2)
    p_map = generate_identity_p_map(chars)
    for i in range(len(cx)):
        p_map[cx[i]] = cx2[i]

    return p_map


def generate_identity_p_map(chars):
    """
    Generates an identity permutation map for given list of characters

    Arguments:
    chars: list of characters

    Returns:
    p_map: an identity permutation map

    """
    p_map = {}
    for c in chars:
        p_map[c] = c

    return p_map


def scramble_text(text, p_map):
    """
    Scrambles a text given a permutation map

    Arguments:
    text: text to scramble, list of characters

    p_map: permutation map to scramble text based upon

    Returns:
    text_2: the scrambled text
    """
    text_2 = []
    for c in text:
        text_2.append(p_map[c])

    return text_2


def shuffle_text(text, i1, i2):
    """
    Shuffles a text given the index from where to shuffle and
    the upto what we should shuffle

    Arguments:
    i1: index from where to start shuffling from

    i2: index upto what we should shuffle, excluded.
    """

    y = text[i1:i2]
    random.shuffle(y)
    t = copy(text)
    t[i1:i2] = y
    return t


def move_one_step(p_map):
    """
    Swaps two characters in the given p_map

    Arguments:
    p_map: A p_map

    Return:
    p_map_2: new p_map, after swapping the characters
    """

    keys = az_list()
    sample = random.sample(keys, 2)
    p_map_2 = deepcopy(p_map)
    p_map_2[sample[1]] = p_map[sample[0]]
    p_map_2[sample[0]] = p_map[sample[1]]

    # NOTE (wshtan 2023-03-22) Below, we try multiple swaps: {{
    #sample = random.sample(keys, 4)
    #p_map_2 = deepcopy(p_map)
    #p_map_2[sample[1]] = p_map[sample[0]]
    #p_map_2[sample[2]] = p_map[sample[1]]
    #p_map_2[sample[3]] = p_map[sample[2]]
    #p_map_2[sample[0]] = p_map[sample[3]]
    # }}
    # NOTE (wshtan 2023-03-22) The code above does not work at all???

    return p_map_2


def pretty_string(text, full=False):
    """
    Pretty formatted string
    """
    if not full:
        return "".join(text[1:200]) + "..."
    else:
        return "".join(text) + "..."


def compute_statistics(filename):
    """
    Returns the statistics for a text file.

    Arguments:
    filename: name of the file

    Returns:
    char_to_ix: mapping from character to index

    ix_to_char: mapping from index to character

    transition_probabilities[i,j]: gives the probability of j following i, smoothed by laplace smoothing

    frequency_statistics[i]: gives number of times character i appears in the document
    """
    data = open(filename, "r").read()  # should be simple plain text file
    chars = list(set(data))
    N = len(chars)
    char_to_ix = {c: i for i, c in enumerate(chars)}
    ix_to_char = {i: c for i, c in enumerate(chars)}

    transition_matrix = np.ones((N, N))
    frequency_statistics = np.zeros(N)
    i = 0
    while i < len(data) - 1:
        c1 = char_to_ix[data[i]]
        c2 = char_to_ix[data[i + 1]]
        transition_matrix[c1, c2] += 1
        frequency_statistics[c1] += 1
        i += 1

    frequency_statistics[c2] += 1
    transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)

    return char_to_ix, ix_to_char, transition_matrix, frequency_statistics


## -- deciphering_utils.py:


def compute_log_probability(
    text, permutation_map, char_to_ix, frequency_statistics, transition_matrix
):
    """
    Computes the log probability of a text under a given permutation map (switching the
    charcter c from permutation_map[c]), given the text statistics

    Note: This is quite slow, as it goes through the whole text to compute the probability,
    if you need to compute the probabilities frequently, see compute_log_probability_by_counts.

    Arguments:
    text: text, list of characters

    permutation_map[c]: gives the character to replace 'c' by

    char_to_ix: characters to index mapping

    frequency_statistics: frequency of character i is stored in frequency_statistics[i]

    transition_matrix: probability of j following i

    Returns:
    p: log likelihood of the given text
    """
    t = text
    p_map = permutation_map
    cix = char_to_ix
    fr = frequency_statistics
    tm = transition_matrix

    i0 = cix[p_map[t[0]]]
    p = np.log(fr[i0])
    i = 0
    while i < len(t) - 1:
        subst = p_map[t[i + 1]]
        i1 = cix[subst]
        p += np.log(tm[i0, i1])
        i0 = i1
        i += 1

    return p


def compute_transition_counts(text, char_to_ix):
    """
    Computes transition counts for a given text, useful to compute if you want to compute
    the probabilities again and again, using compute_log_probability_by_counts.

    Arguments:
    text: Text as a list of characters

    char_to_ix: character to index mapping

    Returns:
    transition_counts: transition_counts[i, j] gives number of times character j follows i
    """
    N = len(char_to_ix)
    transition_counts = np.zeros((N, N))
    c1 = text[0]
    i = 0
    while i < len(text) - 1:
        c2 = text[i + 1]
        transition_counts[char_to_ix[c1], char_to_ix[c2]] += 1
        c1 = c2
        i += 1

    return transition_counts


def compute_log_probability_by_counts(
    transition_counts,
    text,
    permutation_map,
    char_to_ix,
    frequency_statistics,
    transition_matrix,
):
    """
    Computes the log probability of a text under a given permutation map (switching the
    charcter c from permutation_map[c]), given the transition counts and the text

    Arguments:

    transition_counts: a matrix such that transition_counts[i, j] gives the counts of times j follows i,
                       see compute_transition_counts

    text: text to compute probability of, should be list of characters

    permutation_map[c]: gives the character to replace 'c' by

    char_to_ix: characters to index mapping

    frequency_statistics: frequency of character i is stored in frequency_statistics[i]

    transition_matrix: probability of j following i stored at [i, j] in this matrix

    Returns:

    p: log likelihood of the given text
    """
    #debug("compute_log_probability_by_counts(): hello")
    #debug("compute_log_probability_by_counts(): permutation_map: ", permutation_map)
    c0 = char_to_ix[permutation_map[text[0]]]
    p = np.log(frequency_statistics[c0])

    p_map_indices = {}
    for c1, c2 in iter(permutation_map.items()):
        p_map_indices[char_to_ix[c1]] = char_to_ix[c2]

    indices = [value for (key, value) in sorted(p_map_indices.items())]

    #debug("compute_log_probability_by_counts(): transition_counts: ", repr(transition_counts))
    #debug("compute_log_probability_by_counts(): transition_matrix: ", repr(transition_matrix))
    p += np.sum(transition_counts * np.log(transition_matrix[indices, :][:, indices]))

    return p


def compute_difference(text_1, text_2):
    """
    Compute the number of times to text differ in character at same positions

    Arguments:

    text_1: first text list of characters
    text_2: second text, should have same length as text_1

    Returns
    cnt: number of times the texts differ in character at same positions
    """
    cnt = 0
    for x, y in zip(text_1, text_2):
        if y != x:
            cnt += 1

    return cnt


def get_state(text, transition_matrix, frequency_statistics, char_to_ix):
    """
    Generates a default state of given text statistics

    Arguments:
    pretty obvious

    Returns:
    state: A state that can be used along with,
           compute_probability_of_state, propose_a_move,
           and pretty_state for metropolis_hastings

    """
    transition_counts = compute_transition_counts(text, char_to_ix)
    p_map = generate_identity_p_map(char_to_ix.keys())

    state = {
        "text": text,
        "transition_matrix": transition_matrix,
        "frequency_statistics": frequency_statistics,
        "char_to_ix": char_to_ix,
        "permutation_map": p_map,
        "transition_counts": transition_counts,
    }

    return state


def compute_probability_of_state(state):
    """
    Computes the probability of given state using compute_log_probability_by_counts
    """

    p = compute_log_probability_by_counts(
        state["transition_counts"],
        state["text"],
        state["permutation_map"],
        state["char_to_ix"],
        state["frequency_statistics"],
        state["transition_matrix"],
    )

    return p


def propose_a_move(state):
    """
    Proposes a new move for the given state,
    by moving one step (randomly swapping two characters)
    """
    new_state = {}
    for key, value in iter(state.items()):
        new_state[key] = value
    new_state["permutation_map"] = move_one_step(state["permutation_map"])
    return new_state


def pretty_state(state, full=False):
    """
    Returns the state in a pretty format
    """
    if not full:
        return pretty_string(
            scramble_text(state["text"][1:200], state["permutation_map"]), full
        )
    else:
        return pretty_string(
            scramble_text(state["text"], state["permutation_map"]), full
        )


## -- metropolis_hastings.py:


def metropolis_hastings(
    initial_state,
    proposal_function,
    log_density,
    iters=100,
    print_every=10,
    tolerance=0.02,
    error_function=None,
    pretty_state=None,
):
    """
    Runs a metropolis hastings algorithm given the settings

    Arguments:

    initial_state: state from where we should start moving

    proposal_function: proposal function for next state, it takes the current state
                       and returns the next state

    log_density: log probability(upto an unknown normalization constant) function, takes a
                 state as input, and gives the log(probability*some constant) of the state.

    iters: number of iters to continue

    print_every: print every $ iterations the current statistics. For diagnostics purposes.

    tolerance: if acceptance rate drops below this, we stop the simulation

    error_function: computes the error for current state. Printed every print_every iterations.
                    Just for your diagnostics.

    pretty_state: A function from your side to print the current state in a pretty format.

    Returns:

    states: List of states generated during simulation

    cross_entropies: list of negative log probabilites during the simulation.

    errors: lists of errors generated if given error_function, none otherwise.

    """

    p1 = log_density(initial_state)
    errors = []
    cross_entropies = []

    state = initial_state
    cnt = 0
    accept_cnt = 0
    error = -1
    states = [initial_state]
    it = 0

    #while it < iters:
    for i in range(iters):
        # propose a move
        new_state = proposal_function(state)
        p2 = log_density(new_state)

        u = random.random()

        # accept the new move with probability p2-p1
        if p2 - p1 > np.log(u):
            # increment the acceptance counter
            accept_cnt += 1
            debug(f"metropolis_hastings(): acc / iter: [{accept_cnt} / {i}]")
            debug(f"metropolis_hastings(): new score: {p2}")
            debug(f"metropolis_hastings(): new_state: ", repr(new_state))
            debug("metropolis_hastings(): text: ", pretty_state(new_state)[:128])

            # update the state
            state = new_state

            # update the current state probability
            p1 = p2

            # append errors and states
            cross_entropies.append(p1)
            states.append(state)
            if error_function is not None:
                error = error_function(state)
                errors.append(error)

            # print if required
            #if it % print_every == 0:
            #    acceptance = float(accept_cnt) / float(cnt)
            #    s = ""
            #    if pretty_state is not None:
            #        s = "Current state : " + pretty_state(state)

            #    print(
            #        f"[{it} / {iters}]: Entropy : ",
            #        -p1,
            #        ", Error : ",
            #        error,
            #        ", Acceptance : ",
            #        acceptance,
            #    )
            #    print(s)

            #    if acceptance < tolerance:
            #        break

            #    #cnt = 0
            #    #accept_cnt = 0

    if error_function is None:
        errors = None

    return states, cross_entropies, errors


def main(argv):
    inputfile = None
    decodefile = None
    parser = OptionParser()

    parser.add_option(
        "-i", "--input", dest="inputfile", help="input file to train the code on"
    )

    parser.add_option(
        "-d", "--decode", dest="decode", help="file that needs to be decoded"
    )

    parser.add_option(
        "-e",
        "--iters",
        dest="iterations",
        help="number of iterations to run the algorithm for",
        default=5000,
    )

    parser.add_option(
        "-t",
        "--tolerance",
        dest="tolerance",
        help="percentate acceptance tolerance, before we should stop",
        default=0.02,
    )

    parser.add_option(
        "-p",
        "--print_every",
        dest="print_every",
        help="number of steps after which diagnostics should be printed",
        default=500,
    )

    (options, args) = parser.parse_args(argv)

    filename = options.inputfile
    decode = options.decode
    learning_material_size = os.path.getsize(filename) # in bytes

    if filename is None:
        print("Input file is not specified. Type -h for help.")
        sys.exit(2)

    if decode is None:
        print("Decoding file is not specified. Type -h for help.")
        sys.exit(2)

    print("main(): computing statistics...")
    learning_started_at = time.time()
    char_to_ix, ix_to_char, tr, fr = compute_statistics(filename)
    learning_ended_at = time.time()
    learning_time_used = learning_ended_at - learning_started_at
    print(f"main(): computing statistics... Done. Duration: {learning_time_used}")

    s = list(open(decode, "r").read())
    scrambled_text = list(s)
    #i = 0
    initial_state = get_state(scrambled_text, tr, fr, char_to_ix)
    states = []
    entropies = []

    #while i < 3:  #{{

    iters = int(options.iterations)
    print_every = int(options.print_every)
    tolerance = float(options.tolerance)
    decipher_started_at = time.time()
    print("main(): start decipher. time: {decipher_started_at}")
    state, lps, _ = metropolis_hastings(
        initial_state,
        propose_a_move,
        compute_probability_of_state,
        iters=iters,
        print_every=print_every,
        tolerance=tolerance,
        pretty_state=pretty_state,
    )
    states.extend(state)
    entropies.extend(lps)
    decipher_ended_at = time.time()
    decipher_time_used = decipher_ended_at - decipher_started_at

    #}}
    #i += 1

    p = list(zip(states, entropies))
    p.sort(key=lambda x: x[1])

    print("main(): Best Guesses : ")
    for j in range(1, 6):
        print(pretty_state(p[-j][0], full=True)[:128])
        print("****")

    print("\nmain(): Benchmark:\n")
    print(f"    - time used for learning       : {learning_time_used * 1000} [ms]")
    print(f"    - learing material size        : {learning_material_size} [byte]")
    print(f"    - time used for learning per kb: {learning_time_used*1000*1024/ learning_material_size} [ms / kilobyte]")
    print(f"    - time used for decipher       : {decipher_time_used * 1000} [ms]")
    print(f"    - iteration count              : {iters}")
    print(f"    - time used for each iteration : {decipher_time_used * 1000 / iters} [ms / iteration]")


if __name__ == "__main__":
    main(sys.argv)