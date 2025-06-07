#include "inverse.h"

void mat_minors(Matrix *dst, const Matrix *mat) {
    if (dst->rows != mat->rows || dst->cols != mat->cols) {
        return;
    }

    if (!isSquare(mat)) {
        return;
    }

    for (int r = 0; r < mat->rows; r++) {
        for (int c = 0; c < mat->cols; c++) {
            Matrix *submatrix = createMatrix(mat->rows - 1, mat->cols - 1);
            int dr = 0;
            for (int r1 = 0; r1 < submatrix->rows; r1++) {
                int dc = 0;
                if (r1 == r) {
                    dr = 1;
                }
                for (int c1 = 0; c1 < submatrix->cols; c1++) {
                    if (c1 == c) {
                        dc = 1;
                    }
                    submatrix->data[r1][c1] = mat->data[r1 + dr][c1 + dc];
                }
            }
            dst->data[r][c] = mat_determinant(submatrix, NULL);
            destroyMatrix(submatrix);
        }
    }
}

void mat_cofactors(Matrix *dst, const Matrix *minors) {
    if (dst->rows != minors->rows || dst->cols != minors->cols) {
        return;
    }

    if (!isSquare(minors)) {
        return;
    }

    for (int r = 0; r < minors->rows; r++) {
        for (int c = 0; c < minors->cols; c++) {
            dst->data[r][c] = minors->data[r][c] * ((r + c) % 2 == 0 ? 1 : -1);
        }
    }
}

double mat_determinant(const Matrix *mat, const Matrix *cofactors) {
    if (mat->rows == 1 && mat->cols == 1) {
        return mat->data[0][0];
    }

    if (!isSquare(mat)) {
        printf("Matrix is not square\n");
        return 0;
    }

    const Matrix *mcofactors = cofactors;
    Matrix *mcf = NULL;

    if (!mcofactors) {
        mcf = createMatrix(mat->rows, mat->cols);
        Matrix *minors = createMatrix(mat->rows, mat->cols);
        mat_minors(minors, mat);
        mat_cofactors(mcf, minors);
        destroyMatrix(minors);
        mcofactors = mcf;
    } else {
        if (mat->rows != cofactors->rows || mat->cols != cofactors->cols) {
            return 0;
        }
    }
    double det = 0;
    for (int c = 0; c < mat->cols; c++) {
        det += mat->data[0][c] * mcofactors->data[0][c];
    }
    if (mcf) {
        destroyMatrix(mcf);
    }
    return det;
}

double mat_invert(Matrix *dst, const Matrix *mat, Matrix *minors, Matrix *cofactors) {
    // only square matrices allowed
    if (mat->rows != mat->cols) {
        return 0;
    }

    // determine matrix of minors
    Matrix *mminors = minors ? minors : createMatrix(mat->rows, mat->cols);
    mat_minors(mminors, mat);

    // determine matrix of cofactors
    Matrix *mcofactors = cofactors ? cofactors : createMatrix(mat->rows, mat->cols);
    mat_cofactors(mcofactors, mminors);

    double det = mat_determinant(mat, mcofactors);

    // transpose cofactors to get adjugate
    transpose(mcofactors, mcofactors);

    multiplyScalar(dst, mcofactors, 1 / det);

    // transpose adjugate to get back cofactors
    transpose(mcofactors, mcofactors);

    // if the caller didn't provide destination matrices, destroy the created ones
    if (!minors) {
        destroyMatrix(mminors);
    }
    if (!cofactors) {
        destroyMatrix(mcofactors);
    }
    return det;
}