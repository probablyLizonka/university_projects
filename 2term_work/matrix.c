#include "matrix.h"

Matrix *createMatrix(int rows, int cols) {
    Matrix *mat = malloc(sizeof(Matrix));
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (double **) malloc(rows * sizeof(double *));

    for (int i = 0; i < rows; i++) {
        mat->data[i] = (double *) malloc(cols * sizeof(double));
    }
    return mat;
}

void makeIdentity(Matrix *mat) {
    if (mat->rows != mat->cols) {
        return;
    }
    for (int r = 0; r < mat->rows; r++) {
        for (int c = 0; c < mat->cols; c++) {
            mat->data[r][c] = r == c;
        }
    }
}

void zeroMatrix(Matrix *mat) {
    for (int r = 0; r < mat->rows; r++) {
        for (int c = 0; c < mat->cols; c++) {
            mat->data[r][c] = 0;
        }
    }
}

Matrix *createzeroMatrix(int rows, int cols) {
    Matrix *mat = createMatrix(rows, cols);
    zeroMatrix(mat);
    return mat;
}

Matrix *createIdentityMatrix(int size) {
    Matrix *mat = createMatrix(size, size);
    makeIdentity(mat);
    return mat;
}

Matrix *createMatrixWithElements(int rows, int cols, ...) {
    Matrix *mat = createMatrix(rows, cols);

    int r = 0, c = 0;
    int count = rows * cols;

    va_list arglist;
    va_start(arglist, cols);

    for (int i = 0; i < count; i++) {
        mat->data[r][c] = va_arg(arglist, int);
        if (++c >= cols) {
            c = 0;
            r++;
        }
    }

    va_end(arglist);

    return mat;
}

Matrix *createMatrixWithElementsFrom1D(int rows, int cols, double *elements) {
    Matrix *mat = createMatrix(rows, cols);
    for (int r = 0; r < rows; r++) {
        memcpy(mat->data[r], elements + r * cols, cols * sizeof(double));
    }
    return mat;
}

Matrix *createMatrixWithElementsFrom2D(int rows, int cols, double **elements) {
    Matrix *mat = createMatrix(rows, cols);
    for (int r = 0; r < rows; r++) {
        memcpy(mat->data[r], elements[r], cols * sizeof(double ));
    }
    return mat;
}

void copyEntries(Matrix *dst, const Matrix *src) {
    if (dst->rows != src->rows || dst->cols != src->cols) {
        return;
    }
    for (int r = 0; r < src->rows; r++) {
        memcpy(dst->data[r], src->data[r], src->cols * sizeof(double));
    }
}

Matrix *copyMatrix(const Matrix *mat) {
    Matrix *copy = createMatrix(mat->rows, mat->cols);
    copyEntries(copy, mat);
    return copy;
}

void destroyMatrix(Matrix *mat) {
    for (int r = 0; r < mat->rows; r++) {
        free(mat->data[r]);
    }
    free(mat->data);
    free(mat);
}

void transpose(Matrix *dst, const Matrix *mat) {
    if (dst->cols != mat->rows || dst->rows != mat->cols) {
        return;
    }
    if (mat->rows == mat->cols) {
        for (int r = 0; r < mat->rows; r++) {
            dst->data[r][r] = mat->data[r][r];
            for (int c = r + 1; c < mat->cols; c++) {
                double toSwap = mat->data[c][r];
                dst->data[c][r] = mat->data[r][c];
                dst->data[r][c] = toSwap;
            }
        }
    } else {
        for (int r = 0; r < mat->rows; r++) {
            for (int c = 0; c < mat->cols; c++) {
                dst->data[c][r] = mat->data[r][c];
            }
        }
    }
}

int isZero(const Matrix *mat) {
    for (int r = 0; r < mat->rows; r++) {
        for (int c = 0; c < mat->cols; c++) {
            if (mat->data[r][c] != 0) {
                return 0;
            }
        }
    }
    return 1;
}

int isIdentity(const Matrix *mat) {
    if (mat->rows != mat->cols) {
        return 0;
    }
    for (int r = 0; r < mat->rows; r++) {
        for (int c = 0; c < mat->cols; c++) {
            if (mat->data[r][c] != (r == c)) {
                return 0;
            }
        }
    }
    return 1;
}

int isSquare(const Matrix *m) {
    return m->rows == m->cols;
}
