#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <immintrin.h>

using namespace std;

// 3. Оптимизация: векторизация с использованием AVX
void dgemm_opt_3(int n, const vector<vector<double>>& A, const vector<vector<double>>& B, 
    vector<vector<double>>& C, double alpha = 1.0, double beta = 0.0) {
    constexpr int simd_width = 4; // для AVX (4 double за раз)

    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            __m256d a_ik_vec = _mm256_set1_pd(alpha * A[i][k]);
            int j = 0;
            for (; j <= n - simd_width; j += simd_width) {
                __m256d b_kj_vec = _mm256_loadu_pd(&B[k][j]);
                __m256d c_ij_vec = _mm256_loadu_pd(&C[i][j]);
                __m256d res = _mm256_fmadd_pd(a_ik_vec, b_kj_vec, c_ij_vec);
                _mm256_storeu_pd(&C[i][j], res);
            }
            for (; j < n; ++j) {
                C[i][j] += alpha * A[i][k] * B[k][j];
            }
        }
    }
    if (beta != 1.0) {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                C[i][j] *= beta;
    }
}

int main(int argc, char* argv[]) {
    auto begin = std::chrono::steady_clock::now();
    if (argc < 2) {
        cout << "Использование: " << argv[0] << " <размер матрицы> [<размер блока>]\n";
        return 1;
    }

    int n = atoi(argv[1]);
    int block_size = (argc >= 3) ? atoi(argv[2]) : 32;

    if (n <= 0 || block_size <= 0) {
        cout << "Ошибка: размер матрицы и блока должны быть положительными числами.\n";
        return 1;
    }

    srand(static_cast<unsigned>(time(nullptr)));

    vector<vector<double>> A(n, vector<double>(n));
    vector<vector<double>> B(n, vector<double>(n));
    vector<vector<double>> C(n, vector<double>(n, 0.0));

    // Заполнение случайными числами
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            A[i][j] = static_cast<double>(rand()) / RAND_MAX;
            B[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }

    dgemm_opt_3(n, A, B, C);
    auto end = chrono::steady_clock::now();
    auto elapsed_ns = chrono::duration_cast<chrono::duration<double>>(end - begin);
    cout << "Time: " << elapsed_ns.count() << " sec" << endl;
    return 0;
}
