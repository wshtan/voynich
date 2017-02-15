// decipherer.cpp


//| Quick start:
//|
//| To compile, type:
//|
//|     g++ -o ./decipherer.app ./decipherer.cpp
//|
//| To run, type:
//|
//|     ./decipherer.app


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>


namespace mcmc {


#define URAND()    ((double)(rand()) / (double)(RAND_MAX))


// The bigram language model. So far it only supports ascii/latin-1 encoding.
struct LanguageModel {
	int symbol_count;
	int frequency_matrix[256][256];
	int freq_of_symbol[256];

	LanguageModel(FILE *corpus_file);
	double compute_log_score_of(char *text, int len);
};


LanguageModel::LanguageModel(FILE *corpus_file) {
	int i;
	int j;
	char currch; // current symbol
	char nextch; // next symbol 
	memset((void *)(this->freq_of_symbol), 0, 256 * sizeof(int));
	memset((void *)(this->frequency_matrix), 0, 256 * 256 * sizeof(int));
	// Read file:
	do {
		currch = fgetc(corpus_file);  // TODO Batch the call.
		if (currch == EOF) { // Empty file:
			break;  // goto count_symbol;
		}
		this->freq_of_symbol[currch] = 1;
		for (;;) {
			nextch = fgetc(corpus_file);
			if (nextch == EOF) {
				break;  // goto count_symbol;
			}
			this->freq_of_symbol[nextch] += 1;
			this->frequency_matrix[currch][nextch] += 1;
			currch = nextch;
		}
	} while (0);
	// count_symbol:
	for (i = 0; i < 256; i++) {
		if (this->freq_of_symbol[i] != 0) {
			this->symbol_count += 1;
		}
	}
}


double LanguageModel::compute_log_score_of(
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
	printf("Per::Per(): N: %d\n", N);
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
	int selection = URAND() * 52 * 51;
	int first_letter = selection / 52;
	int second_letter = selection % 52;
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
			if (URAND() > 0.5) {
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
	int len
) {
	PermutationMap pm;
	pm.scramble();
	pm.apply_inplace(plain_text_buffer, len);
}


void decipher(
	LanguageModel *language_model,
	char          *scrambled_text,
	int            scrambled_text_length,
	int            iterations
) {
	PermutationMap current_map;  // Init with identity map.
	PermutationMap proposed_map;
	char *proposed_text = (char *)calloc(scrambled_text_length, sizeof(char));
	if (proposed_text == NULL) {
		printf("ERROR: cannot allocate buffer.\n");
		return;
	}
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
				log(URAND()) <= (proposed_log_score - current_log_score)
			)
		);
		if (should_accept) {
			current_map.copy_from(&proposed_map);
			current_log_score = proposed_log_score;
			ac_count += 1;
			printf("decipher(): accepted! ac / it: [%d / %d]\n", ac_count, it_count);
			printf("decipher(): score: %f\n", proposed_log_score);
			printf("decipher(): current text: %s\n", proposed_text);
		}
	}
	free(proposed_text);
	return;
}

} // namespace mcmc


const char CORPUS_FILE_PATH[] = "./data/warpeace_input.txt";


int main() {
	const char text[] = (
		"I dont prefer a coding oriented project. "
		"I prefer the project can be based on the "
		"knowledge learned in this class. I would "
		"rather read new material and summarize it "
		"than solve a problem on my own. Besides, "
		"the format of the project I prefer to write "
		"a report instead of a presentation. "
		"For this class, I hope to explain more example "
		"questions in detail in the lecture to facilitate "
		"the understanding and application of concepts "
		"and theories, and questions in quiz are also "
		"needed to be explained."
	);
	FILE *corpus_file = fopen(CORPUS_FILE_PATH, "r");
	if (corpus_file == NULL) {
		printf("ERROR: cannot open the corpus file: %s\n", CORPUS_FILE_PATH);
	}
	mcmc::LanguageModel english_model(corpus_file);
	printf("Original text: %s\n", text);
	mcmc::scramble_text_inplace((char *)text, strlen(text));
	printf("Scrambled text: %s\n", text);
	mcmc::decipher(&english_model, (char *)text, strlen(text), 30000);
	return 0;
}
