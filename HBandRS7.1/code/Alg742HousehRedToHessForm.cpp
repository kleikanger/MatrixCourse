#include <armadillo.h>
#include <iostream>
#include <math.h>

using namespace std;
using namespace arma;

void house(double & beta, colvec & v, colvec & x) {
    double sigma, mu;

    int n = x.n_rows;

    colvec xspan = x(span(1, n - 1));
    sigma = dot(xspan, xspan);

    v = zeros<colvec > (n);
    v(span(1, n - 1)) = xspan;

    //zero if sigma is zero
    if (sigma == 0) {
        beta = 0;
    } else {
        mu = sqrt(x(0) * x(0) + sigma);

        if (x(0) <= 0) {
            v(0) = x(0) - mu;
        } else {
            v(0) = -sigma / (x(0) + mu);
        }

        beta = 2 / (sigma / (v(0) * v(0)) + 1);
        v /= v(0);
    }
}

void houseRedHessenberg(mat & A) {

    double beta;
    colvec v, x;
    mat ImBvvT;

    int n = A.n_cols;
    mat I = eye<mat > (n - 1, n - 1);

    for (int k = 0; k < n - 2; k++) {
        x = A(span(k + 1, n-1), k);
        house(beta, v, x);
        A(span(k + 1, n - 1), k) = x;

        ImBvvT = I - beta * v * strans(v);
        A(span(k + 1, n - 1), span(k, n - 1)) = ImBvvT * A(span(k + 1, n - 1), span(k, n - 1));
        A(span(), span(k + 1, n - 1)) = A(span(), span(k + 1, n - 1)) * ImBvvT;
    }

}

int main() {

    mat A;
    A << 1 << 5 << 7 << endr
            << 3 << 0 << 6 << endr
            << 4 << 3 << 1 << endr;

    cout << A << endl;
    houseRedHessenberg(A);
    cout << A << endl;

    return 0;
}
