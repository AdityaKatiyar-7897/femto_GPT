#include <stdio.h>
#include <string.h>

#define MAX_VOCAB 32
#define MAX_TOKENS 128
#define WORD_LEN 32

typedef struct {
    char word[WORD_LEN];
} Token;

Token vocab[MAX_VOCAB];
int vocab_size = 0;

int token_ids[MAX_TOKENS];
int token_count = 0;

int transition_counts[MAX_VOCAB][MAX_VOCAB]; 

float probability_table[MAX_VOCAB][MAX_VOCAB];

int get_token_id(const char *word) {
    for (int i = 0; i < vocab_size; i++) {
        if (strcmp(vocab[i].word, word) == 0) {
            return i;
        }
    }

    strcpy(vocab[vocab_size].word, word);
    vocab_size++;
    return vocab_size - 1;
}

void add_sentence(const char *sentence) {
    char buffer[256];
    strcpy(buffer, sentence);

    char *word = strtok(buffer, " ");

    while (word != NULL) {
        int id = get_token_id(word);
        token_ids[token_count++] = id;
        word = strtok(NULL, " ");
    }
}

void build_transition_counts()
{
    for (int i = 0; i < token_count - 1; i++)
    {
        int current = token_ids[i];
        int next = token_ids[i + 1];

        transition_counts[current][next]++;
    }
}

void build_probability_table()
{
   
    for (int from = 0; from < vocab_size; from++)
    {
        int total_transitions = 0;

        for (int to = 0; to < vocab_size; to++)
        {
            total_transitions += transition_counts[from][to];
        }

        if (total_transitions == 0)
        {
            continue; 
        }

        for (int to = 0; to < vocab_size; to++)
        {
            probability_table[from][to] = 
                (float)transition_counts[from][to] / total_transitions;
        }
    }
}

void print_vocab() {
    printf("\nVOCAB TABLE:\n");
    for (int i = 0; i < vocab_size; i++) {
        printf("id %d -> %s\n", i, vocab[i].word);
    }
}

void print_tokens() {
    printf("\nTOKEN IDS:\n");
    for (int i = 0; i < token_count; i++) {
        printf("%d ", token_ids[i]);
    }
    printf("\n");

    printf("\nTOKENS AS WORDS:\n");
    for (int i = 0; i < token_count; i++) {
        printf("%s ", vocab[token_ids[i]].word);
    }
    printf("\n");
}

void print_transition_counts()
{
    printf("\nTRANSITION COUNTS\n\n");

    for (int from = 0; from < vocab_size; from++)
    {
        printf("%s:\n", vocab[from].word);

        for (int to = 0; to < vocab_size; to++)
        {
            int count = transition_counts[from][to];

            if (count > 0)
            {
                printf(
                    "   -> %-10s : %d\n",
                    vocab[to].word,
                    count
                );
            }
        }

        printf("\n");
    }
}

void print_probability_table()
{
    printf("\nPROBABILITY TABLE\n\n");

    for (int from = 0; from < vocab_size; from++)
    {
        printf("%s:\n", vocab[from].word);

        for (int to = 0; to < vocab_size; to++)
        {
            float probability = probability_table[from][to];

            if (probability > 0.0f)
            {
                printf(
                    "   -> %-10s : %.2f\n",
                    vocab[to].word,
                    probability
                );
            }
        }

        printf("\n");
    }
}

int main() {
    add_sentence("cat eats fish");
    add_sentence("dog eats food");
    add_sentence("cat drinks milk");
    add_sentence("dog drinks water");

    print_vocab();
    print_tokens();

    build_transition_counts();

    print_transition_counts();

    build_probability_table();

    print_probability_table();

    return 0;
}
