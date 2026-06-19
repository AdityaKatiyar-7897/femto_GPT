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

int main() {
    add_sentence("cat eats fish");
    add_sentence("dog eats food");
    add_sentence("cat drinks milk");
    add_sentence("dog drinks water");

    print_vocab();
    print_tokens();

    return 0;
}
