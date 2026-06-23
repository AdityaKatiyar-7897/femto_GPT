#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MAX_VOCAB 64
#define MAX_TOKENS 512
#define WORD_LEN 32
#define EMBED_DIM 8
#define HIDDEN_DIM 8

#define MAX_EXAMPLES 1024

#define BLOCK_SIZE 4

typedef struct {
    char word[WORD_LEN];
} Token;

typedef struct
{
    float Q[HIDDEN_DIM];
    float K[HIDDEN_DIM];
    float V[HIDDEN_DIM];
}
QKV;

Token vocab[MAX_VOCAB];
int vocab_size = 0;

int token_ids[MAX_TOKENS];
int token_count = 0;

int transition_counts[MAX_VOCAB][MAX_VOCAB];
float probability_table[MAX_VOCAB][MAX_VOCAB];

float embeddings[MAX_VOCAB][EMBED_DIM];

float Wq[EMBED_DIM][HIDDEN_DIM];
float Wk[EMBED_DIM][HIDDEN_DIM];
float Wv[EMBED_DIM][HIDDEN_DIM];

float Wout[HIDDEN_DIM][MAX_VOCAB];

float clip(float x, float limit)
{
    if (x >  limit) return  limit;
    if (x < -limit) return -limit;
    return x;
}

void softmax(float values[], int count)
{
    float max = values[0];

    for(int i = 1; i < count; i++)
    {
        if(values[i] > max)
            max = values[i];
    }

    float sum = 0.0f;

    for(int i = 0; i < count; i++)
    {
        values[i] = expf(values[i] - max);
        sum += values[i];
    }

    for(int i = 0; i < count; i++)
    {
        values[i] /= sum;
    }
}

int get_token_id(const char *word)
{
    for (int i = 0; i < vocab_size; i++)
    {
        if (strcmp(vocab[i].word, word) == 0)
        {
            return i;
        }
    }

    strcpy(vocab[vocab_size].word, word);
    vocab_size++;

    return vocab_size - 1;
}

void add_sentence(const char *sentence)
{
    token_ids[token_count++] = get_token_id("<BOS>");

    char buffer[256];
    strcpy(buffer, sentence);

    char *word = strtok(buffer, " ");

    while (word != NULL)
    {
        token_ids[token_count++] = get_token_id(word);
        word = strtok(NULL, " ");
    }

    token_ids[token_count++] = get_token_id("<EOS>");
}

void print_vocab()
{
    printf("\nVOCAB TABLE:\n");

    for (int i = 0; i < vocab_size; i++)
    {
        printf("id %d -> %s\n", i, vocab[i].word);
    }
}

void print_tokens()
{
    printf("\nTOKEN IDS:\n");

    for (int i = 0; i < token_count; i++)
    {
        printf("%d ", token_ids[i]);
    }

    printf("\n\nTOKENS AS WORDS:\n");

    for (int i = 0; i < token_count; i++)
    {
        printf("%s ", vocab[token_ids[i]].word);
    }

    printf("\n");
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
                printf("    -> %-10s : %d\n",
                       vocab[to].word,
                       count);
            }
        }

        printf("\n");
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
                printf("    -> %-10s : %.2f\n",
                       vocab[to].word,
                       probability);
            }
        }

        printf("\n");
    }
}

int sample_next_token(int current_token)
{
    float r = (float)rand() / (float)RAND_MAX;
    float cumulative = 0.0f;

    for (int next = 0; next < vocab_size; next++)
    {
        cumulative += probability_table[current_token][next];

        if (r <= cumulative)
        {
            return next;
        }
    }

    return -1;
}

void debug_distribution(int token_id)
{
    printf("\nCurrent token: %s\n", vocab[token_id].word);

    for (int i = 0; i < vocab_size; i++)
    {
        float p = probability_table[token_id][i];

        if (p > 0.0f)
        {
            printf("%s -> %.2f\n", vocab[i].word, p);
        }
    }
}

void generate_text(const char *start_word, int max_steps)
{
    int current = get_token_id(start_word);

    printf("\nGenerated:\n");
    printf("%s ", vocab[current].word);

    for (int step = 0; step < max_steps; step++)
    {
        debug_distribution(current);

        int next = sample_next_token(current);

        if (next == -1)
        {
            break;
        }

        if (strcmp(vocab[next].word, "<EOS>") == 0)
        {
            break;
        }

        printf("%s ", vocab[next].word);

        current = next;
    }

    printf("\n");
}

void generate_from_bos(int max_steps)
{
    int current = get_token_id("<BOS>");

    printf("\nGenerated from <BOS>:\n");

    for (int step = 0; step < max_steps; step++)
    {
        debug_distribution(current);

        int next = sample_next_token(current);

        if (next == -1)
        {
            break;
        }

        if (strcmp(vocab[next].word, "<EOS>") == 0)
        {
            break;
        }

        printf("%s ", vocab[next].word);

        current = next;
    }

    printf("\n");
}

void initialize_embeddings()
{
    for (int token = 0; token < vocab_size; token++)
    {
        for (int dim = 0; dim < EMBED_DIM; dim++)
        {
            embeddings[token][dim] =
                (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * 0.1f;
        }
    }
}

void print_embeddings()
{
    printf("\nEMBEDDINGS\n\n");

    for (int token = 0; token < vocab_size; token++)
    {
        printf("%-10s : ", vocab[token].word);

        for (int dim = 0; dim < EMBED_DIM; dim++)
        {
            printf("%.3f ", embeddings[token][dim]);
        }

        printf("\n");
    }
}

void random_matrix(float matrix[EMBED_DIM][HIDDEN_DIM])
{
    for (int row = 0; row < EMBED_DIM; row++)
    {
        for (int col = 0; col < HIDDEN_DIM; col++)
        {
            matrix[row][col] =
                (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * 0.1f;
        }
    }
}

void project(
    float input[EMBED_DIM],
    float matrix[EMBED_DIM][HIDDEN_DIM],
    float output[HIDDEN_DIM]
)
{
    for (int col = 0; col < HIDDEN_DIM; col++)
    {
        output[col] = 0.0f;

        for (int row = 0; row < EMBED_DIM; row++)
        {
            output[col] += input[row] * matrix[row][col];
        }
    }
}

void build_qkv(int token_id)
{
    float Q[HIDDEN_DIM];
    float K[HIDDEN_DIM];
    float V[HIDDEN_DIM];

    project(embeddings[token_id], Wq, Q);
    project(embeddings[token_id], Wk, K);
    project(embeddings[token_id], Wv, V);

    printf("\nTOKEN: %s\n", vocab[token_id].word);

    printf("\nQ:\n");
    for (int i = 0; i < HIDDEN_DIM; i++)
    {
        printf("%.3f ", Q[i]);
    }

    printf("\n\nK:\n");
    for (int i = 0; i < HIDDEN_DIM; i++)
    {
        printf("%.3f ", K[i]);
    }

    printf("\n\nV:\n");
    for (int i = 0; i < HIDDEN_DIM; i++)
    {
        printf("%.3f ", V[i]);
    }

    printf("\n");
}

float dot_product(
    float a[HIDDEN_DIM],
    float b[HIDDEN_DIM]
)
{
    float sum = 0.0f;

    for (int i = 0; i < HIDDEN_DIM; i++)
    {
        sum += a[i] * b[i];
    }

    return sum;
}

QKV compute_qkv(int token_id)
{
    QKV result;

    project(
        embeddings[token_id],
        Wq,
        result.Q
    );

    project(
        embeddings[token_id],
        Wk,
        result.K
    );

    project(
        embeddings[token_id],
        Wv,
        result.V
    );

    return result;
}

void initialize_output_matrix()
{
    for (int row = 0; row < HIDDEN_DIM; row++)
    {
        for (int col = 0; col < MAX_VOCAB; col++)
        {
            Wout[row][col] =
                (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * 0.1f;
        }
    }
}

void attention_demo()
{
    int sequence[3];

    sequence[0] = get_token_id("cat");
    sequence[1] = get_token_id("eats");
    sequence[2] = get_token_id("fish");

    QKV qkv[3];

    for (int i = 0; i < 3; i++)
    {
        qkv[i] = compute_qkv(sequence[i]);
    }

    printf("\nSEQUENCE\n\n");

    for (int i = 0; i < 3; i++)
    {
        printf("%s ", vocab[sequence[i]].word);
    }

    printf("\n");

    float scores[3];

    for (int j = 0; j < 3 ; j++)
    {
        scores[j] = dot_product(qkv[0].Q, qkv[j].K);
    }

    printf("\nRAW SCORES\n\n");

    for (int j = 0; j < 3; j++)
    {
        printf(
            "cat -> %-5s : %.3f\n", vocab[sequence[j]].word,scores[j]);

    }
    softmax(scores, 3);

    printf("\nSOFTMAX WEIGHTS\n\n");

    for (int j = 0; j < 3; j++)
    {
        printf("cat -> %-5s : %.3f\n", vocab[sequence[j]].word, scores[j]);
    }

    printf("\nV VECTORS\n\n");

    for (int token = 0; token < 3; token++)
    {
        printf("%s : ", vocab[sequence[token]].word);

        for (int dim = 0; dim < HIDDEN_DIM; dim++)
        {
            printf("%.3f ", qkv[token].V[dim]);
        }

        printf("\n");
    }

    printf("\nATTENTION MIX\n\n");

    for (int token = 0; token < 3; token++)
    {
        printf(
            "%.3f * V(%s)\n",
            scores[token],
            vocab[sequence[token]].word
        );
    }

    float attention_output[HIDDEN_DIM];

    for (int dim = 0 ; dim < HIDDEN_DIM; dim++)
    {
        attention_output[dim] = 0.0f;
    }


    for (int token = 0; token < 3; token ++)
    {
        for (int dim = 0; dim < HIDDEN_DIM; dim++)
        {
            attention_output[dim] += scores[token] * qkv[token].V[dim];
        }
    }

    printf("\nAttention output for 'cat'\n\n");

    for (int dim = 0; dim < HIDDEN_DIM; dim++)
    {
        printf("%.3f ", attention_output[dim]);
    }

    printf("\n");

    float logits[MAX_VOCAB];

    for (int token = 0; token < vocab_size; token++)
    {
        logits[token] = 0.0f;

        for (int dim = 0; dim < HIDDEN_DIM; dim++)
        {
            logits[token] += attention_output[dim] * Wout[dim][token];
        }
    }

    printf("\nLOGITS\n\n");

    for (int token = 0; token < vocab_size; token++)
    {
        printf("%-10s : %.3f\n", vocab[token].word, logits[token]);
    }

    softmax(logits, vocab_size);

    printf("\nNEXT TOKEN PROBABILITES\n\n");

    for (int token = 0; token < vocab_size; token++)
    {
        printf("%-10s : %.3f\n", vocab[token].word, logits[token]);
    }
}

float forward_loss(int context[], int context_len, int target_token)
{
    QKV qkv[8];

    for (int i = 0; i < context_len; i++)
    {
        qkv[i] = compute_qkv(context[i]);
    }

    int query_pos = context_len - 1;

    float scores[8];

    float scale = 1.0f / sqrtf((float)HIDDEN_DIM);

    for (int j = 0; j < context_len; j++)
    {
        scores[j] = dot_product(qkv[query_pos].Q, qkv[j].K) * scale;
    }

    softmax(scores, context_len);

    float attention_output[HIDDEN_DIM];

    for (int dim = 0; dim < HIDDEN_DIM; dim++)
    {
        attention_output[dim] = 0.0f;
    }

    for (int token = 0; token < context_len; token++)
    {
        for (int dim = 0; dim < HIDDEN_DIM; dim++)
        {
            attention_output[dim] += scores[token] * qkv[token].V[dim];
        }
    }

    float logits[MAX_VOCAB];

    for (int token = 0; token < vocab_size; token++)
    {
        logits[token] = 0.0f;

        for (int dim = 0; dim < HIDDEN_DIM; dim++)
        {
            logits[token] += attention_output[dim] * Wout[dim][token];
        }
    }

    softmax(logits, vocab_size);

    float predicted_probability = logits[target_token];
    float loss = -logf(predicted_probability + 1e-9f);

    printf("\nFORWARD LOSS TEST\n\n");

    printf("Context: ");
    for (int i = 0; i < context_len; i++)
    {
        printf("%s ", vocab[context[i]].word);
    }

    printf("\nTarget: %s\n", vocab[target_token].word);

    printf("\nPredicted probability for target: %.6f\n", predicted_probability);
    printf("Loss: %.6f\n", loss);

    printf("\nTop probabilities:\n");
    for (int token = 0; token < vocab_size; token++)
    {
        printf("%-10s : %.4f\n", vocab[token].word, logits[token]);
    }

    return loss;
}

void train_wout_once(int context[], int context_len, int target_token, float learning_rate)
{
    QKV qkv[8];

    for (int i = 0; i < context_len; i++)
    {
        qkv[i] = compute_qkv(context[i]);
    }

    int query_pos = context_len - 1;

    float scores[8];
    float scale = 1.0f / sqrtf((float)HIDDEN_DIM);

    for (int j = 0; j < context_len; j++)
    {
        scores[j] = dot_product(qkv[query_pos].Q, qkv[j].K) * scale;
    }

    softmax(scores, context_len);

    float attention_output[HIDDEN_DIM];

    for (int dim = 0; dim < HIDDEN_DIM; dim++)
    {
        attention_output[dim] = 0.0f;
    }

    for (int token = 0; token < context_len; token++)
    {
        for (int dim = 0; dim < HIDDEN_DIM; dim++)
        {
            attention_output[dim] += scores[token] * qkv[token].V[dim];
        }
    }

    float logits[MAX_VOCAB];

    for (int token = 0; token < vocab_size; token++)
    {
        logits[token] = 0.0f;

        for (int dim = 0; dim < HIDDEN_DIM; dim++)
        {
            logits[token] += attention_output[dim] * Wout[dim][token];
        }
    }

    softmax(logits, vocab_size);

    float d_attention[HIDDEN_DIM];

    for (int dim = 0; dim < HIDDEN_DIM; dim++)
    {
        d_attention[dim] = 0.0f;
    }

    for (int v = 0; v < vocab_size; v++)
    {
        float d_logit = logits[v] - (v == target_token ? 1.0f : 0.0f);

        for (int dim = 0; dim < HIDDEN_DIM; dim++)
        {
            float old_w = Wout[dim][v];

            d_attention[dim] += d_logit * old_w;

            Wout[dim][v] -= learning_rate * clip(
                d_logit * attention_output[dim],
                1.0f
            );
        }
    }

    for (int t = 0; t < context_len; t++)
    {
        for (int in_dim = 0; in_dim < EMBED_DIM; in_dim++)
        {
            for (int out_dim = 0; out_dim < HIDDEN_DIM; out_dim++)
            {
                float d_V = scores[t] * d_attention[out_dim];

                Wv[in_dim][out_dim] -=
                    learning_rate * clip(d_V * embeddings[context[t]][in_dim], 1.0f);
            }
        }
    }

    for (int t = 0; t < context_len; t++)
    {
        for (int in_dim = 0; in_dim < EMBED_DIM; in_dim++)
        {
            float grad = 0.0f;

            for (int out_dim = 0; out_dim < HIDDEN_DIM; out_dim++)
            {
                float d_V = scores[t] * d_attention[out_dim];
                grad += d_V * Wv[in_dim][out_dim];
            }

            embeddings[context[t]][in_dim] -= learning_rate * clip(grad, 1.0f);
        }
    }
}


int predict_next_token(int context[], int context_len)
{
    QKV qkv[8];

    for (int i = 0; i < context_len; i++)
    {
        qkv[i] = compute_qkv(context[i]);
    }

    int query_pos = context_len - 1;

    float scores[8];
    float scale = 1.0f / sqrtf((float)HIDDEN_DIM);

    for (int j = 0; j < context_len; j++)
    {
        scores[j] = dot_product(qkv[query_pos].Q, qkv[j].K) * scale;
    }

    softmax(scores, context_len);

    float attention_output[HIDDEN_DIM];

    for (int dim = 0; dim < HIDDEN_DIM; dim++)
    {
        attention_output[dim] = 0.0f;
    }

    for (int token = 0; token < context_len; token++)
    {
        for (int dim = 0; dim < HIDDEN_DIM; dim++)
        {
            attention_output[dim] += scores[token] * qkv[token].V[dim];
        }
    }

    float logits[MAX_VOCAB];

    for (int token = 0; token < vocab_size; token++)
    {
        logits[token] = 0.0f;

        for (int dim = 0; dim < HIDDEN_DIM; dim++)
        {
            logits[token] += attention_output[dim] * Wout[dim][token];
        }
    }

    softmax(logits, vocab_size);

    int best_token = 0;
    float best_probability = logits[0];

    for (int token = 1; token < vocab_size; token++)
    {
        if (logits[token] > best_probability)
        {
            best_probability = logits[token];
            best_token = token;
        }
    }

    return best_token;
}

int find_token_id(const char *word)
{
    for (int i = 0; i < vocab_size; i++)
    {
        if (strcmp(vocab[i].word, word) == 0)
        {
            return i;
        }
    }

    return -1;
}

void generate(const char *word1, const char *word2, int max_steps)
{
    int id1 = find_token_id(word1);
    int id2 = find_token_id(word2);

    if (id1 == -1 || id2 == -1)
    {
        printf("Unknown word.\n");
        return;
    }

    printf("%s %s ", word1, word2);

    int context[2];
    context[0] = id1;
    context[1] = id2;

    for (int step = 0; step < max_steps; step++)
    {
        int next = predict_next_token(context, 2);

        if (strcmp(vocab[next].word, "<EOS>") == 0) break;

        printf("%s ", vocab[next].word);

        context[0] = context[1];
        context[1] = next;
    }

    printf("\n");
}

void generate_gpt(int max_steps)
{
    int context[BLOCK_SIZE];
    int context_len = 1;

    context[0] = find_token_id("<BOS>");

    printf("\nGenerated from <BOS>:\n");

    for (int step = 0; step < max_steps; step++)
    {
        int next = predict_next_token(context, context_len);

        if (strcmp(vocab[next].word, "<EOS>") == 0)
            break;

        printf("%s ", vocab[next].word);

        if (context_len < BLOCK_SIZE)
        {
            context[context_len] = next;
            context_len++;
        }
        else
        {
            for (int i = 0; i < BLOCK_SIZE - 1; i++)
            {
                context[i] = context[i + 1];
            }

            context[BLOCK_SIZE - 1] = next;
        }
    }

    printf("\n");
}


int main()
{
    srand(time(NULL));

    // eats patterns — multiple subjects, consistent targets force generalization
    add_sentence("cat eats fish");
    add_sentence("cat eats meat");
    add_sentence("dog eats meat");
    add_sentence("dog eats food");
    add_sentence("fox eats chicken");
    add_sentence("fox eats meat");
    add_sentence("bird eats seeds");
    add_sentence("cow eats grass");
    add_sentence("boy eats apple");
    add_sentence("girl eats apple");
    add_sentence("boy eats food");

    // drinks patterns
    add_sentence("cat drinks milk");
    add_sentence("dog drinks water");
    add_sentence("bird drinks water");
    add_sentence("cow drinks water");
    add_sentence("fox drinks water");
    add_sentence("boy drinks milk");
    add_sentence("girl drinks water");

    // likes patterns
    add_sentence("cat likes fish");
    add_sentence("dog likes bones");
    add_sentence("fish likes water");
    add_sentence("bird likes seeds");

    // motion patterns
    add_sentence("fox runs fast");
    add_sentence("dog runs fast");
    add_sentence("boy runs fast");
    add_sentence("bird flies sky");
    add_sentence("fish swims water");

    // is patterns
    add_sentence("apple is red");
    add_sentence("grass is green");
    add_sentence("sky is blue");
    add_sentence("water is blue");

    // chaining sentences so generation can go beyond 1 step
    add_sentence("cat eats fish fast");
    add_sentence("dog eats meat fast");
    add_sentence("fox runs fast food");
    add_sentence("bird flies sky blue");
    add_sentence("cow eats grass green");
    add_sentence("boy eats apple red");
    add_sentence("girl drinks water blue");

    initialize_embeddings();

    random_matrix(Wq);
    random_matrix(Wk);
    random_matrix(Wv);

    initialize_output_matrix();

    int contexts[MAX_EXAMPLES][BLOCK_SIZE];
    int context_lens[MAX_EXAMPLES];
    int targets[MAX_EXAMPLES];
    int example_count = 0;

    for (int start = 0; start < token_count; start++)
    {
        if (strcmp(vocab[token_ids[start]].word, "<BOS>") != 0)
            continue;

        for (int end = start; end < token_count - 1; end++)
        {
            int context_len = end - start + 1;

            if (context_len > BLOCK_SIZE)
                break;

            int target = token_ids[end + 1];

            for (int i = 0; i < context_len; i++)
            {
                contexts[example_count][i] = token_ids[start + i];
            }

            if (example_count >= MAX_EXAMPLES)
            {
                printf("MAX_EXAMPLES exceeded\n");
                exit(1);
            }

            context_lens[example_count] = context_len;
            targets[example_count] = target;
            example_count++;

            if (strcmp(vocab[target].word, "<EOS>") == 0)
                break;
        }
    }

    printf("Training examples: %d\n", example_count);
    printf("Vocab size: %d\n\n", vocab_size);

    printf("Training");
    fflush(stdout);

    for (int epoch = 0; epoch < 25000; epoch++)
    {
        for (int example = 0; example < example_count; example++)
        {
            train_wout_once(
                contexts[example],
                context_lens[example],
                targets[example],
                0.0005f
            );
        }

        if (epoch % 2000 == 0)
        {
            float total_loss = 0.0f;

            for (int e = 0; e < example_count; e++)
            {
                QKV qkv[8];

                for (int i = 0; i < context_lens[e]; i++)
                {
                    qkv[i] = compute_qkv(contexts[e][i]);
                }

                int query_pos = context_lens[e] - 1;

                float scores[8];
                float scale = 1.0f / sqrtf((float)HIDDEN_DIM);

                for (int j = 0; j < context_lens[e]; j++)
                {
                    scores[j] = dot_product(qkv[query_pos].Q, qkv[j].K) * scale;
                }

                softmax(scores, context_lens[e]);

                float attn[HIDDEN_DIM];
                for (int d = 0; d < HIDDEN_DIM; d++) attn[d] = 0.0f;
                for (int t = 0; t < context_lens[e]; t++)
                    for (int d = 0; d < HIDDEN_DIM; d++)
                        attn[d] += scores[t] * qkv[t].V[d];

                float logits[MAX_VOCAB];
                for (int v = 0; v < vocab_size; v++)
                {
                    logits[v] = 0.0f;
                    for (int d = 0; d < HIDDEN_DIM; d++)
                        logits[v] += attn[d] * Wout[d][v];
                }
                softmax(logits, vocab_size);
                total_loss += -logf(logits[targets[e]] + 1e-9f);
            }

            printf("  epoch %5d  loss %.4f\n", epoch, total_loss / example_count);
            fflush(stdout);
        }
    }

    printf("Training examples: %d\n", example_count);
    printf("MAX_EXAMPLES: %d\n", MAX_EXAMPLES);

    printf("\ngeneration after training \n\n");

    generate("cat",  "eats",   8);
    generate("dog",  "drinks", 8);
    generate("bird", "flies",  8);
    generate("fox",  "runs",   8);
    generate("cow",  "eats",   8);
    generate("sky",  "is",     8);
    generate("boy",  "eats",   8);

    printf("\nGPT generation \n");
    generate_gpt(12);
    generate_gpt(12);
    generate_gpt(12);

    printf("\n Lets test :\n");
    printf("Enter two words to generate from, or q q to quit.\n\n");

    while (1)
    {
        char word1[32];
        char word2[32];

        printf("> ");
        fflush(stdout);

        if (scanf("%31s %31s", word1, word2) != 2) break;

        if (strcmp(word1, "q") == 0 && strcmp(word2, "q") == 0) break;

        int id1 = find_token_id(word1);
        int id2 = find_token_id(word2);

        if (id1 == -1 || id2 == -1)
        {
            printf("Unknown word. Vocab: ");
            for (int i = 0; i < vocab_size; i++)
                printf("%s ", vocab[i].word);
            printf("\n");
            continue;
        }

        generate(word1, word2, 8);
    }





    return 0;
}
