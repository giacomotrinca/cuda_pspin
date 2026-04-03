// ===========================================================================
// Test 1: Quartet combinatorial index round-trip
//
//  q → (ii,jj,kk,ll) → q'   must give  q' == q   for all q in [0, C(N,4))
//
// Encoding (sorted ii < jj < kk < ll):
//   q = C(ll,4) + C(kk,3) + C(jj,2) + ii
// where C(n,r) = n*(n-1)*...*(n-r+1)/r!
//
// GPU kernel checks every q in parallel; a single failure sets a flag.
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Encode sorted (ii < jj < kk < ll) → linear quartet index
__device__ __host__ long long encode_quartet(int ii, int jj, int kk, int ll) {
    return (long long)ll*(ll-1)*(ll-2)*(ll-3)/24
         + (long long)kk*(kk-1)*(kk-2)/6
         + (long long)jj*(jj-1)/2
         + ii;
}

// Decode linear index q → sorted (ii, jj, kk, ll)
__device__ __host__ void decode_quartet(long long q, int* ii, int* jj, int* kk, int* ll) {
    int l = 3;
    while ((long long)l*(l-1)*(l-2)*(l-3)/24 <= q) l++;
    l--;
    long long rem = q - (long long)l*(l-1)*(l-2)*(l-3)/24;
    int k = 2;
    while ((long long)k*(k-1)*(k-2)/6 <= rem) k++;
    k--;
    rem -= (long long)k*(k-1)*(k-2)/6;
    int j = 1;
    while ((long long)j*(j-1)/2 <= rem) j++;
    j--;
    rem -= (long long)j*(j-1)/2;
    *ii = (int)rem;
    *jj = j;
    *kk = k;
    *ll = l;
}

// Kernel: each thread checks one q value
__global__ void check_roundtrip_kernel(int N, long long nq, int* d_fail) {
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;

    int ii, jj, kk, ll;
    decode_quartet(q, &ii, &jj, &kk, &ll);

    // Check ordering
    if (ii < 0 || jj <= ii || kk <= jj || ll <= kk || ll >= N) {
        atomicExch(d_fail, 1);
        return;
    }

    // Re-encode
    long long q2 = encode_quartet(ii, jj, kk, ll);
    if (q2 != q) {
        atomicExch(d_fail, 1);
    }
}

int main(int argc, char** argv) {
    // Test N values from 4 to N_max (default 20)
    int N_max = 20;
    if (argc > 1) N_max = atoi(argv[1]);
    if (N_max < 4) N_max = 4;

    printf("=== Test: quartet index round-trip  N = 4 .. %d ===\n\n", N_max);

    int all_pass = 1;
    int* d_fail;
    CUDA_CHECK(cudaMalloc(&d_fail, sizeof(int)));

    for (int N = 4; N <= N_max; N++) {
        long long nq = (long long)N*(N-1)*(N-2)*(N-3)/24;

        CUDA_CHECK(cudaMemset(d_fail, 0, sizeof(int)));

        int block = 256;
        int grid  = (int)((nq + block - 1) / block);
        check_roundtrip_kernel<<<grid, block>>>(N, nq, d_fail);
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_fail = 0;
        CUDA_CHECK(cudaMemcpy(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost));

        if (h_fail) {
            printf("  N=%2d  C(N,4)=%8lld  FAIL\n", N, nq);
            all_pass = 0;
        } else {
            printf("  N=%2d  C(N,4)=%8lld  PASS\n", N, nq);
        }
    }

    // Also test exhaustive CPU encode→decode→encode for small N
    printf("\n--- CPU exhaustive check N=4..8 ---\n");
    for (int N = 4; N <= 8; N++) {
        long long nq = (long long)N*(N-1)*(N-2)*(N-3)/24;
        int fail = 0;
        long long q_enum = 0;
        for (int l = 3; l < N && !fail; l++)
            for (int k = 2; k < l && !fail; k++)
                for (int j = 1; j < k && !fail; j++)
                    for (int i = 0; i < j && !fail; i++, q_enum++) {
                        long long q = encode_quartet(i, j, k, l);
                        if (q != q_enum) { fail = 1; break; }
                        int ii, jj, kk, ll;
                        decode_quartet(q, &ii, &jj, &kk, &ll);
                        if (ii != i || jj != j || kk != k || ll != l) { fail = 1; break; }
                    }
        if (fail || q_enum != nq) {
            printf("  N=%d  FAIL\n", N);
            all_pass = 0;
        } else {
            printf("  N=%d  all %lld quartets  PASS\n", N, nq);
        }
    }

    CUDA_CHECK(cudaFree(d_fail));

    printf("\n%s\n", all_pass ? "ALL PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
