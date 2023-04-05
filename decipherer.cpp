// decipherer.cpp


//| Quick start:
//|
//| To compile, type:
//|
//|     g++ -o ./a.out ./decipherer.cpp
//|
//| To run, type:
//|
//|     ./a.out


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>


namespace mcmc {


inline double urand() {
	// Pick a number from Unif((0, 1)):
	return ((double)(rand()) / (double)(RAND_MAX));
}


// The bigram language model. So far it only supports latin-1/ascii charset (and encoding).
struct BigramLanguageModel {
	int frequency_matrix[256][256];

	BigramLanguageModel(FILE *corpus_file);
	double compute_log_score_of(char *text, int len);
};


BigramLanguageModel::BigramLanguageModel(FILE *corpus_file) {
	int i;
	int j;
	char currch; // current symbol
	char nextch; // next symbol 
	memset((void *)(this->frequency_matrix), 0, 256 * 256 * sizeof(int));
	// Read file:
	currch = fgetc(corpus_file);
	if (currch == EOF) { // Empty file:
		return;
	}
	for (;;) {
		nextch = fgetc(corpus_file);
		if (nextch == EOF) {
			return;
		}
		this->frequency_matrix[currch][nextch] += 1;
		currch = nextch;
	}
}


double BigramLanguageModel::compute_log_score_of(
	char *text,
	int   len
) {
	double score = 0.0;
	char currch;
	char nextch;
	int freq;
	int i;
	for (i = 0; i < len; i++) {
		currch = text[i];
		nextch = text[i + 1];
		freq = this->frequency_matrix[currch][nextch];
		if (freq != 0) {
			score += log(freq);
		} else {
			// Ignore it.
		}
	}
	return score;
}


struct PermutationMap {
	char m[256];

	PermutationMap();
	void copy_from(PermutationMap *src);
	void swap_two_letters_at_random();  // random transposition
	void scramble();
	void apply(char *dst, char *src, int len);
	void apply_inplace(char *buf, int len);
};


PermutationMap::PermutationMap() {
	int i;
	const int N = 256;
	for (i = 0; i < N; i++) {
		this->m[i] = (char)i;
	}
}


void PermutationMap::copy_from(PermutationMap *src) {
	int i;
	const int N = 256;
	for (i = 0; i < N; i++) {
		this->m[i] = src->m[i];
	}
}


void PermutationMap::swap_two_letters_at_random() {
	int selection = urand() * 52 * 51;
	int first_letter = selection / 52;
	int second_letter = selection % 51;
	if (second_letter >= first_letter) {
		second_letter += 1;
	}
	int first_letter_index = (
		(first_letter < 26) ? (
			// uppercase:
			'A' + first_letter
		) : (
			// lowercase:
			'a' + first_letter - 26
		)
	);
	int second_letter_index = (
		(second_letter < 26) ? (
			// uppercase:
			'A' + second_letter
		) : (
			// lowercase:
			'a' + second_letter - 26
		)
	);
	char tmp = this->m[first_letter_index];
	this->m[first_letter_index] = this->m[second_letter_index];
	this->m[second_letter_index] = tmp;
}


void PermutationMap::scramble() {
	char letters[52];
	char tmp;
	int i;
	int j;
	for (i = 0; i < 26; i++) {
		letters[i] = 'A' + (char)(i);
		letters[26 + i] = 'a' + (char)(i);
	}
	// Suffle the `letters` array:
	for (i = 0; i < 52; i++) {
		for (j = 0; j < 52; j++) {
			if (urand() > 0.5) {
				// Swap:
				tmp = letters[i];
				letters[i] = letters[j];
				letters[j] = tmp;
			}
		}
	}
	// Update the map:
	for (i = 0; i < 26; i++) {
		this->m['A' + (char)i] = letters[i];
		this->m['a' + (char)i] = letters[26 + i];
	}
	return;
}


void PermutationMap::apply(char *dst, char *src, int len) {
	int i;
	for (i = 0; i < len; i++) {
		dst[i] = this->m[src[i]];
	}
}


void PermutationMap::apply_inplace(char *buf, int len) {
	this->apply(buf, buf, len);
}


void scramble_text_inplace(
	char *plain_text_buffer,
	int   len
) {
	PermutationMap pm;
	pm.scramble();
	pm.apply_inplace(plain_text_buffer, len);
}


void decipher(
	BigramLanguageModel *language_model,
	char                *scrambled_text,
	int                  scrambled_text_length,
	char                *result_buffer,
	int                  iterations
) {
	PermutationMap current_map;  // Init with identity map.
	PermutationMap proposed_map;
	char *proposed_text = result_buffer;
	double current_log_score = language_model->compute_log_score_of(scrambled_text, scrambled_text_length);
	double proposed_log_score = 0.0;
	int it_count = 0;
	int ac_count = 0;
	bool should_accept;
	for (it_count = 1; it_count <= iterations; it_count ++) {
		proposed_map.copy_from(&current_map);
		proposed_map.swap_two_letters_at_random();
		proposed_map.apply(proposed_text, scrambled_text, scrambled_text_length);
		proposed_log_score = language_model->compute_log_score_of(proposed_text, scrambled_text_length);
		should_accept = (
			(proposed_log_score > current_log_score) ? true : (
				log(urand()) <= (proposed_log_score - current_log_score)
			)
		);
		if (should_accept) {
			current_map.copy_from(&proposed_map);
			current_log_score = proposed_log_score;
			ac_count += 1;
			//printf("decipher(): accepted! ac / it: [%d / %d]\n", ac_count, it_count);
			//printf("decipher(): score: %f\n", proposed_log_score);
			//printf("decipher(): current text: %s\n", proposed_text);
		}
	}
	return;
}

} // namespace mcmc


const char ENGLISH_CORPUS_FILE_PATH[] = "./data/warpeace_input.txt";


int main() {
	char text[] = (
		"I arrive now at the ineffable core of my story. And here begins my despair as a writer. All language is a set of symbols whose use among its speakers assumes a shared past. How, then, can I translate into words the limitless Aleph, which my floundering mind can scarcely encompass?"
	);
	const int textlen = strlen(text);
	char result_buffer[sizeof(text)] = {0};
	FILE *english_corpus_file = fopen(ENGLISH_CORPUS_FILE_PATH, "r");
	if (english_corpus_file == NULL) {
		printf("ERROR: cannot open the corpus file: %s\n", ENGLISH_CORPUS_FILE_PATH);
	}
	mcmc::BigramLanguageModel english_model(english_corpus_file);
	printf("Original text: %s\n\n", text);
	mcmc::scramble_text_inplace((char *)text, textlen);
	printf("Scrambled text: %s\n\n", text);
	mcmc::decipher(&english_model, (char *)text, textlen, result_buffer, 30000);
	printf("Final result: %s\n\n", result_buffer);
	return 0;
}
