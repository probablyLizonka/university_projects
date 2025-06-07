#include "arithmetic.h"

void add(Matrix *dst, const Matrix *m1, const Matrix *m2, int *F) {
    if (m1->rows != m2->rows || m1->cols != m2->cols) {
        printf("Wrong size\n");
        *F = 1;
        return;
    }
    for (int r = 0; r < m1->rows; r++) {
        for (int c = 0; c < m1->rows; c++) {
            dst->data[r][c] = m1->data[r][c] + m2->data[r][c];
        }
    }
}

void multiplyScalar(Matrix *dst, const Matrix *mat, double scale) {
    for (int r = 0; r < mat->rows; r++) {
        for (int c = 0; c < mat->cols; c++) {
            dst->data[r][c] = mat->data[r][c] * scale;
        }
    }
}

void multiplyMatrix(Matrix *dst, const Matrix *m1, const Matrix *m2, int *F) {
    // matrices cannot be multiplied
    if (m1->cols != m2->rows) {
        printf("Wrong size\n");
        *F = 1;
        return;
    }
    // destination matrix is of the wrong size
    if (dst->rows != m1->rows || dst->cols != m2->cols) {
        return;
    }
    for (int r = 0; r < dst->rows; r++) {
        for (int c = 0; c < dst->cols; c++) {
            double sum = 0;
            for (int i = 0; i < m1->cols; i++) {
                sum += m1->data[r][i] * m2->data[i][c];
            }
            dst->data[r][c] = sum;
        }
    }
}