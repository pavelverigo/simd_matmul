#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define ASSERT(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "Assertion failed: \"%s\" (file: %s, function: %s, line: %d)\n", #cond, __FILE__, __FUNCTION__, __LINE__); \
            abort(); \
        } \
    } while(0)

#include <time.h>
void clock_print_elapsed_s(clock_t start) {
    clock_t end = clock();
    float elapsed = (end - start) / (CLOCKS_PER_SEC * 1.0f);
    printf("elapsed: %.6f\n", elapsed);
}

typedef struct {
    uint8_t *buffer;
    int n;
} u8_array;

u8_array u8_array_load_from_file(const char *file_name, int n) {
    u8_array res;
    res.n = n;

    FILE *file = fopen(file_name, "rb");
    ASSERT(file != NULL);

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    ASSERT(file_size == n);

    res.buffer = malloc(file_size);
    fread(res.buffer, 1, file_size, file);
    fclose(file);

    return res;
}

u8_array u8_array_init(int n) {
    ASSERT(n > 0);
    u8_array res;
    res.buffer = malloc(n * sizeof(uint8_t));
    res.n = n;
    return res;
}

void u8_array_free(u8_array a) {
    free(a.buffer);
}

float u8_array_avg_diff(u8_array a, u8_array b) {
    ASSERT(a.n == b.n);
    float sum = 0.0f;
    for (int i = 0; i < a.n; i++) {
        if (a.buffer[i] == b.buffer[i]) sum += 1.0f; 
    }
    return sum / a.n;
}

typedef struct {
    float *buffer;
    int n;
    int m;
} array_2d;

array_2d a2d_init(int n, int m) {
    ASSERT(n > 0 && m > 0);
    array_2d res;
    res.buffer = malloc(n * m * sizeof(float));
    res.n = n;
    res.m = m;
    return res;
}

void a2d_print(array_2d a) {
    printf("array_2d %dx%d\n", a.n, a.m);
    for (int i = 0; i < a.n; i++) {
        for (int j = 0; j < a.m; j++) {
            printf("%.3f ", a.buffer[a.m * i + j]);
        }
        printf("\n");
    }
}

void a2d_set_value(array_2d a, float value) {
    int num_elements = a.n * a.m;
    for (int i = 0; i < num_elements; i++) a.buffer[i] = value;
}

void a2d_free(array_2d a) {
    free(a.buffer);
}

void a2d_a2d_mul_naive(array_2d dest, array_2d a, array_2d b) {
    ASSERT(dest.n == a.n && dest.m == b.m && a.m == b.n);

    for (int p = 0; p < a.m; p++) {
        for (int i = 0; i < dest.n; i++) {
            for (int j = 0; j < dest.m; j++) {
                dest.buffer[i * dest.m + j] += a.buffer[a.m * i + p] * b.buffer[p * b.m + j];
            }
        }
    }
}

#include <immintrin.h>

#define REGS_A 3
#define REGS_B 4
void matmul_dot_inner(int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
    __m256 csum[REGS_A][REGS_B] = {{0.0f}};
    for (int p = 0; p < k; p++) {
        // Perform the DOT product.
        for (int bi = 0; bi < REGS_B; bi++) {
            __m256 bb = _mm256_loadu_ps(&b[p * ldb + bi * 8]);
            for (int ai = 0; ai < REGS_A; ai++) {
                __m256 aa = _mm256_broadcast_ss(&a[ai * lda + p]);
                csum[ai][bi] = _mm256_fmadd_ps(aa, bb, csum[ai][bi]);
            }
        }
    }

    for (int ai = 0; ai < REGS_A; ai++) {
        for (int bi = 0; bi < REGS_B; bi++) {
            _mm256_storeu_ps(&c[ai * ldc + bi * 8], csum[ai][bi]);
        }
    }
}

void matmul_out_bounds(int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc, int rest_i, int rest_j) {
    for (int p = 0; p < k; p++) {
        for (int i = 0; i < rest_i; i++) {
            for (int j = 0; j < rest_j; j++) {
                c[i * ldc + j] += a[lda * i + p] * b[p * ldb + j];
            }
        }
    }
}

void a2d_a2d_mul_fast(array_2d dest, array_2d a, array_2d b) {
    ASSERT(dest.n == a.n && dest.m == b.m && a.m == b.n);

    int n = dest.n;
    int m = dest.m;
    int k = a.m;

    for (int i = 0; i + REGS_A - 1 < n; i += REGS_A) {
        for (int j = 0; j + REGS_B * 8 - 1 < m; j += REGS_B * 8) {
            matmul_dot_inner(k, &a.buffer[i * a.m], a.m, &b.buffer[j], b.m, &dest.buffer[i * m + j], m);
        }
    }

    int leftover_i = n % REGS_A;
    int leftover_j = m % (8 * REGS_B);
    int i = n - leftover_i;
    int j = m - leftover_j;
    if (leftover_i > 0) {
        matmul_out_bounds(k, &a.buffer[i * a.m], a.m, &b.buffer[0], b.m, &dest.buffer[i * m + 0], m, leftover_i, j);
    }
    if (leftover_j > 0) {
        matmul_out_bounds(k, &a.buffer[0], a.m, &b.buffer[j], b.m, &dest.buffer[j], m, i, leftover_j);
    }
    if (leftover_i > 0 && leftover_j > 0) {
        matmul_out_bounds(k, &a.buffer[i * a.m], a.m, &b.buffer[j], b.m, &dest.buffer[i * m + j], m, leftover_i, leftover_j);
    }
}
#undef REGS_B
#undef REGS_A

void a2d_relu(array_2d dest, array_2d a) {
    ASSERT(dest.n == a.n && dest.m == a.m);
    for (int i = 0; i < a.n; i++) {
        for (int j = 0; j < a.m; j++) {
            int pos = i * a.m + j;
            if (a.buffer[pos] < 0.0f) {
                dest.buffer[pos] = 0.0f;
            } else {
                dest.buffer[pos] = a.buffer[pos];
            }
        }
    }
}

void a2d_argmax_row(u8_array dest, array_2d a) {
    ASSERT(dest.n == a.n);
    for (int i = 0; i < a.n; i++) {
        float best = a.buffer[i * a.m];
        int ind = 0;
        for (int j = 1; j < a.m; j++) {
            if (a.buffer[i * a.m + j] > best) {
                best = a.buffer[i * a.m + j];
                ind = j;
            }
        }
        dest.buffer[i] = ind;
    }
}

array_2d a2d_load_from_file_u8_to_f32(const char *file_name, int n, int m) {
    array_2d res;
    res.n = n;
    res.m = m;
    int num_elements = n * m;
    u8_array tmp = u8_array_load_from_file(file_name, num_elements);

    res.buffer = malloc(num_elements * sizeof(float));
    for (int i = 0; i < num_elements; i++) res.buffer[i] = tmp.buffer[i] / 256.0f;
    return res;
}

array_2d a2d_load_from_file(const char *file_name, int n, int m) {
    array_2d res;
    res.n = n;
    res.m = m;
    int num_elements = n * m;
    u8_array tmp = u8_array_load_from_file(file_name, num_elements * sizeof(float));

    res.buffer = (float *) tmp.buffer;
    return res;
}

#define TRAIN_SZ 60000
#define TEST_SZ 10000

#define IMAGE_SZ 28*28
#define INNER_SZ 128
#define OUT_SZ 10

void forward() {
    array_2d x_train = a2d_load_from_file_u8_to_f32("datasets/X_train.bin", TRAIN_SZ, IMAGE_SZ);
    u8_array y_train = u8_array_load_from_file("datasets/Y_train.bin", TRAIN_SZ);
    printf("x_train %d %d\n", x_train.n, x_train.m);
    printf("y_train %d\n", y_train.n);

    array_2d l1 = a2d_load_from_file("weights/l1.bin", IMAGE_SZ, INNER_SZ);
    array_2d l2 = a2d_load_from_file("weights/l2.bin", INNER_SZ, OUT_SZ);
    printf("l1 %d %d\n", l1.n, l1.m);
    printf("l2 %d %d\n", l2.n, l2.m);

    array_2d tmp1 = a2d_init(TRAIN_SZ, INNER_SZ);
    array_2d tmp2 = a2d_init(TRAIN_SZ, INNER_SZ);
    array_2d tmp3 = a2d_init(TRAIN_SZ, OUT_SZ);
    u8_array res = u8_array_init(TRAIN_SZ);

    // forward
    clock_t start_time = clock();

    printf("1\n");
    a2d_a2d_mul_fast(tmp1, x_train, l1);
    clock_print_elapsed_s(start_time);
    printf("2\n");
    a2d_relu(tmp2, tmp1);
    clock_print_elapsed_s(start_time);
    printf("3\n");
    a2d_a2d_mul_fast(tmp3, tmp2, l2);
    clock_print_elapsed_s(start_time);
    printf("4\n");
    a2d_argmax_row(res, tmp3);
    clock_print_elapsed_s(start_time);
    printf("5\n");
    float avg = u8_array_avg_diff(res, y_train);
    clock_print_elapsed_s(start_time);
    printf("6\n");

    clock_print_elapsed_s(start_time);

    printf("forward %.7f\n", avg);

    a2d_free(x_train);
    u8_array_free(y_train);
    a2d_free(l1);
    a2d_free(l2);
    a2d_free(tmp1);
    a2d_free(tmp2);
    a2d_free(tmp3);
    u8_array_free(res);
}

int main() {
    forward();
}