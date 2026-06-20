#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_VOCAB 32
#define MAX_TOKENS 128
#define WORD_LEN 32
#define EMBED_DIM 4

#define HIDDEN_DIM 3

typedef struct {
  char word[WORD_LEN];
} Token;

Token vocab[MAX_VOCAB];
int vocab_size = 0;

int token_ids[MAX_TOKENS];
int token_count = 0;

int transition_counts[MAX_VOCAB][MAX_VOCAB];
float probability_table[MAX_VOCAB][MAX_VOCAB];
float embeddings[MAX_VOCAB][EMBED_DIM];

float W[EMBED_DIM][HIDDEN_DIM]; //weight matrix

void initialize_weights()
{
    for (int row = 0; row < EMBED_DIM; row++)
    {
        for (int col = 0; col < HIDDEN_DIM; col++)
        {
            W[row][col] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
}

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
  token_ids[token_count++] = get_token_id("<BOS>");

  char buffer[256];
  strcpy(buffer, sentence);

  char *word = strtok(buffer, " ");

  while (word != NULL) {
    token_ids[token_count++] = get_token_id(word);
    word = strtok(NULL, " ");
  }

  token_ids[token_count++] = get_token_id("<EOS>");
}

void build_transition_counts() {
  for (int i = 0; i < token_count - 1; i++) {
    int current = token_ids[i];
    int next = token_ids[i + 1];

    transition_counts[current][next]++;
  }
}

void build_probability_table() {
  for (int from = 0; from < vocab_size; from++) {
    int total_transitions = 0;

    for (int to = 0; to < vocab_size; to++) {
      total_transitions += transition_counts[from][to];
    }

    if (total_transitions == 0) {
      continue;
    }

    for (int to = 0; to < vocab_size; to++) {
      probability_table[from][to] =
          (float)transition_counts[from][to] / total_transitions;
    }
  }
}

void initialize_embeddings() {
  for (int token = 0; token < vocab_size; token++) {
    for (int dim = 0; dim < EMBED_DIM; dim++) {
      embeddings[token][dim] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
  }
}

int sample_next_token(int current_token) {
  float r = (float)rand() / (float)RAND_MAX;

  float cumulative = 0.0f;

  for (int next = 0; next < vocab_size; next++) {
    cumulative += probability_table[current_token][next];

    if (r <= cumulative) {
      return next;
    }
  }

  return -1;
}

/*void print_vocab() {
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

void print_transition_counts() {
  printf("\nTRANSITION COUNTS\n\n");

  for (int from = 0; from < vocab_size; from++) {
    printf("%s:\n", vocab[from].word);

    for (int to = 0; to < vocab_size; to++) {
      int count = transition_counts[from][to];

      if (count > 0) {
        printf("   -> %-10s : %d\n", vocab[to].word, count);
      }
    }

    printf("\n");
  }
}

void print_probability_table() {
  printf("\nPROBABILITY TABLE\n\n");

  for (int from = 0; from < vocab_size; from++) {
    printf("%s:\n", vocab[from].word);

    for (int to = 0; to < vocab_size; to++) {
      float probability = probability_table[from][to];

      if (probability > 0.0f) {
        printf("   -> %-10s : %.2f\n", vocab[to].word, probability);
      }
    }

    printf("\n");
  }
  }

void print_embeddings() {

    printf("\nEMBEDDINGS\n\n");

    for(int token = 0; token < vocab_size; token++)
    {
        printf("%-10s : ", vocab[token].word);

        for (int dim = 0; dim < EMBED_DIM; dim++)
        {
            printf("%.3f ", embeddings[token][dim]);
        }

        printf("\n");
    }
}

void debug_distribution(int token_id) {
  printf("\nCurrent token: %s\n", vocab[token_id].word);

  for (int i = 0; i < vocab_size; i++) {
    float p = probability_table[token_id][i];

    if (p > 0.0f) {
      printf("%s -> %.2f\n", vocab[i].word, p);
    }
  }
}
*/


void generate_text(const char *start_word, int max_steps) {
  int current = get_token_id(start_word);

  printf("\nGenerated:\n");
  printf("%s ", vocab[current].word);

  for (int step = 0; step < max_steps; step++) {
   // debug_distribution(current);

    int next = sample_next_token(current);

    if (next == -1) {
      break;
    }

    if (strcmp(vocab[next].word, "<EOS>") == 0) {
      break;
    }

    printf("%s ", vocab[next].word);

    current = next;
  }

  printf("\n");
}

void generate_from_bos(int max_steps) {
  int current = get_token_id("<BOS>");

  printf("\nGenerated from <BOS>:\n");

  for (int step = 0; step < max_steps; step++) {
    //debug_distribution(current);

    int next = sample_next_token(current);

    if (next == -1) {
      break;
    }

    if (strcmp(vocab[next].word, "<EOS>") == 0) {
      break;
    }

    printf("%s ", vocab[next].word);

    current = next;
  }

  printf("\n");
}

void print_weight_matrix()
{
    printf("\nWEIGHT MATRIX\n\n");

    for (int row = 0; row < EMBED_DIM; row++)
    {
        for (int col = 0; col < HIDDEN_DIM; col++)
        {
            printf("%8.3f ", W[row][col]);
        }

        printf("\n");
    }
}

void project_token(int token_id)
{
    float output[HIDDEN_DIM];

    for (int col = 0; col < HIDDEN_DIM; col++)
    {
        output[col] = 0.0f;

        for (int row = 0; row < EMBED_DIM; row++)
        {
            output[col] +=
                embeddings[token_id][row]
                *
                W[row][col];
        }
    }

    printf("\nTOKEN: %s\n",
           vocab[token_id].word);

    printf("Embedding:\n");

    for (int i = 0; i < EMBED_DIM; i++)
    {
        printf("%.3f ",
               embeddings[token_id][i]);
    }

    printf("\n");

    printf("Projected:\n");

    for (int i = 0; i < HIDDEN_DIM; i++)
    {
        printf("%.3f ",
               output[i]);
    }

    printf("\n");
}

void print_matmul_breakdown(int token_id)
{
    printf("\nMATMUL BREAKDOWN FOR %s\n\n",
           vocab[token_id].word);

    for (int col = 0; col < HIDDEN_DIM; col++)
    {
        printf("output[%d] = ", col);

        for (int row = 0; row < EMBED_DIM; row++)
        {
            printf(
                "(%.2f * %.2f)",
                embeddings[token_id][row],
                W[row][col]
            );

            if (row != EMBED_DIM - 1)
            {
                printf(" + ");
            }
        }

        printf("\n");
    }
}

int main() {
  srand(time(NULL));
  add_sentence("cat eats fish");
  add_sentence("dog eats food");
  add_sentence("cat drinks milk");
  add_sentence("dog drinks water");

  //print_vocab();
  //print_tokens();

  build_transition_counts();

 // print_transition_counts();

  build_probability_table();

  //print_probability_table();

  initialize_embeddings();

 //print_embeddings();
 //
 //

  generate_text("cat", 10);
  generate_text("dog", 10);
  generate_from_bos(10);


  initialize_weights();

  print_weight_matrix();

  project_token(get_token_id("cat"));

  project_token(get_token_id("dog"));

  print_matmul_breakdown(
      get_token_id("cat")
  );

  return 0;
}
