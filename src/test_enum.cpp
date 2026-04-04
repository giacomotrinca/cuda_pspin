// CPU-only test: verify that the 3-type enumeration produces
// exactly the same set of quartets as brute-force iteration over C(N,4).
// Compile:  g++ -O2 -o test_enum test_enum.cpp && ./test_enum

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <set>
#include <tuple>
#include <cassert>

typedef std::tuple<int,int,int,int> Q4;

// Brute-force: enumerate all C(N,4) quartets, keep those involving i0 or j0
static std::set<Q4> brute_force(int N, int i0, int j0) {
    std::set<Q4> result;
    for (int a = 0; a < N; a++)
        for (int b = a+1; b < N; b++)
            for (int c = b+1; c < N; c++)
                for (int d = c+1; d < N; d++) {
                    if (a == i0 || b == i0 || c == i0 || d == i0 ||
                        a == j0 || b == j0 || c == j0 || d == j0)
                        result.insert(Q4(a,b,c,d));
                }
    return result;
}

// 3-type enumeration (same logic as mc_sweep_kernel in mc.cu)
static std::set<Q4> three_type(int N, int i0, int j0) {
    std::set<Q4> result;
    int Nm2 = N - 2;
    long long n_type12 = (long long)Nm2*(Nm2-1)*(Nm2-2)/6;
    long long n_type3  = (long long)Nm2*(Nm2-1)/2;
    long long n_h4     = 2*n_type12 + n_type3;

    for (long long t = 0; t < n_h4; t++) {
        int idx[4];

        if (t < n_type12) {
            // Type 1: {i0, a, b, c}
            long long tt = t;
            int cc = (int)cbrt(6.0*(double)tt);
            if (cc < 2) cc = 2;
            while ((long long)cc*(cc-1)*(cc-2)/6 > tt) cc--;
            while ((long long)(cc+1)*cc*(cc-1)/6 <= tt) cc++;
            long long rem = tt - (long long)cc*(cc-1)*(cc-2)/6;
            int bb = (int)sqrt(2.0*(double)rem);
            if (bb < 1) bb = 1;
            while ((long long)bb*(bb-1)/2 > rem) bb--;
            while ((long long)(bb+1)*bb/2 <= rem) bb++;
            int aa = (int)(rem - (long long)bb*(bb-1)/2);
            int a = aa; if (a >= i0) a++; if (a >= j0) a++;
            int b = bb; if (b >= i0) b++; if (b >= j0) b++;
            int c = cc; if (c >= i0) c++; if (c >= j0) c++;
            idx[0] = i0; idx[1] = a; idx[2] = b; idx[3] = c;
        } else if (t < 2*n_type12) {
            // Type 2: {j0, a, b, c}
            long long tt = t - n_type12;
            int cc = (int)cbrt(6.0*(double)tt);
            if (cc < 2) cc = 2;
            while ((long long)cc*(cc-1)*(cc-2)/6 > tt) cc--;
            while ((long long)(cc+1)*cc*(cc-1)/6 <= tt) cc++;
            long long rem = tt - (long long)cc*(cc-1)*(cc-2)/6;
            int bb = (int)sqrt(2.0*(double)rem);
            if (bb < 1) bb = 1;
            while ((long long)bb*(bb-1)/2 > rem) bb--;
            while ((long long)(bb+1)*bb/2 <= rem) bb++;
            int aa = (int)(rem - (long long)bb*(bb-1)/2);
            int a = aa; if (a >= i0) a++; if (a >= j0) a++;
            int b = bb; if (b >= i0) b++; if (b >= j0) b++;
            int c = cc; if (c >= i0) c++; if (c >= j0) c++;
            idx[0] = j0; idx[1] = a; idx[2] = b; idx[3] = c;
        } else {
            // Type 3: {i0, j0, a, b}
            long long tt = t - 2*n_type12;
            int bb = (int)sqrt(2.0*(double)tt);
            if (bb < 1) bb = 1;
            while ((long long)bb*(bb-1)/2 > tt) bb--;
            while ((long long)(bb+1)*bb/2 <= tt) bb++;
            int aa = (int)(tt - (long long)bb*(bb-1)/2);
            int a = aa; if (a >= i0) a++; if (a >= j0) a++;
            int b = bb; if (b >= i0) b++; if (b >= j0) b++;
            idx[0] = i0; idx[1] = j0; idx[2] = a; idx[3] = b;
        }

        // Sort
        if (idx[0] > idx[1]) { int tmp = idx[0]; idx[0] = idx[1]; idx[1] = tmp; }
        if (idx[2] > idx[3]) { int tmp = idx[2]; idx[2] = idx[3]; idx[3] = tmp; }
        if (idx[0] > idx[2]) { int tmp = idx[0]; idx[0] = idx[2]; idx[2] = tmp; }
        if (idx[1] > idx[3]) { int tmp = idx[1]; idx[1] = idx[3]; idx[3] = tmp; }
        if (idx[1] > idx[2]) { int tmp = idx[1]; idx[1] = idx[2]; idx[2] = tmp; }

        Q4 q(idx[0], idx[1], idx[2], idx[3]);

        // Check for duplicates within the 3-type enumeration
        if (result.count(q)) {
            printf("  DUPLICATE at t=%lld: (%d,%d,%d,%d)\n", t, idx[0], idx[1], idx[2], idx[3]);
        }
        result.insert(q);
    }
    return result;
}

int main() {
    int all_pass = 1;

    for (int N = 4; N <= 20; N++) {
        int n_pairs_tested = 0;
        int n_fail = 0;
        for (int i0 = 0; i0 < N; i0++) {
            for (int j0 = i0+1; j0 < N; j0++) {
                auto bf = brute_force(N, i0, j0);
                auto tt = three_type(N, i0, j0);

                if (bf != tt) {
                    n_fail++;
                    if (n_fail <= 3) {
                        printf("  N=%d i0=%d j0=%d:  brute=%zu  three_type=%zu\n",
                               N, i0, j0, bf.size(), tt.size());
                        // Show differences
                        for (auto& q : bf) {
                            if (!tt.count(q))
                                printf("    MISSING from 3-type: (%d,%d,%d,%d)\n",
                                       std::get<0>(q), std::get<1>(q),
                                       std::get<2>(q), std::get<3>(q));
                        }
                        for (auto& q : tt) {
                            if (!bf.count(q))
                                printf("    EXTRA in 3-type: (%d,%d,%d,%d)\n",
                                       std::get<0>(q), std::get<1>(q),
                                       std::get<2>(q), std::get<3>(q));
                        }
                    }
                }
                n_pairs_tested++;
            }
        }
        printf("N=%2d  pairs=%3d  failures=%d  %s\n",
               N, n_pairs_tested, n_fail, n_fail ? "FAIL" : "PASS");
        if (n_fail) all_pass = 0;
    }

    printf("\n%s\n", all_pass ? "ALL PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
