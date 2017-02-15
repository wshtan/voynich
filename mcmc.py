# The Good Sequence Problem (in the textbook)

import random


class GoodSequenceWalker:
    def __init__(self, m):
        self.m = m
        initial_state = [0 for _ in range(m)]
        self.current_state = initial_state

    def walk(self):
        i = int(random.random() * self.m)
        if self.current_state[i] == 1:
            self.current_state[i] = 0
        else:  # if self.current_state[i] == 0:
            if not (
                (i - 1 >= 0 and self.current_state[i - 1] == 1)
                or (i + 1 < self.m and self.current_state[i + 1] == 1)
            ):
                # can switch from 0 to 1:
                self.current_state[i] = 1


def mean(arr):
    return sum(arr) / len(arr)


def main():
    walker = GoodSequenceWalker(100)
    # walk many steps:
    states = []
    for _ in range(1_000_000):
        walker.walk()
        states.append(sum(walker.current_state))

    print(mean(states))


if __name__ == "__main__":
    main()
