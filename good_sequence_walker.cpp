// good_sequence_walker.cpp - Rewrite mcmc.py using C++ (C-style).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>


struct GoodSequenceWalker {
	int m;
	int *current_state;
	// Methods:
	GoodSequenceWalker(int m);
	~GoodSequenceWalker();
	void walk();
	int get_current_state_sum();
};


GoodSequenceWalker::GoodSequenceWalker(int m) {
	this->m = m;
	// Set initial state:
	this->current_state = (int *)calloc(m, sizeof(int));
}


GoodSequenceWalker::~GoodSequenceWalker() {
	free(this->current_state);
}


void GoodSequenceWalker::walk() {
	// Pick i from Unif({0, 1, ..., this->m - 1}):
	double u = ((double)(rand()) / (double)(RAND_MAX));
	int i = u * this->m;
	if (this->current_state[i] == 1) {
		this->current_state[i] = 0;
	} else {
		if (!(
			(i-1 >= 0      && this->current_state[i-1] == 1) ||
			(i+1 < this->m && this->current_state[i+1] == 1)
		)) {
			// ...
			this->current_state[i] = 1;
		}
	}
}


int GoodSequenceWalker::get_current_state_sum() {
	int i;
	int sum = 0;
	for (i = 0; i < this->m; i++) {
		sum += this->current_state[i];
	}
	return sum;
}


int main() {
	const int N = 1000000;
	int state[N] = {0};
	GoodSequenceWalker walker(100);
	int i;
	int j;
	double dsum;
	// Initialize random number generator:
	srand(time(NULL));
	// Simulate 1 realization with N steps:
	for (i = 0; i < N; i++) {
		walker.walk();
		state[i] = walker.get_current_state_sum();
	}
	// Compute averge (can be done smarter):
	dsum = 0;
	for (i = 0; i < N; i++) {
		dsum += state[i];
	}
	// Print result:
	printf("%f", dsum / N);
	return 0;
}
