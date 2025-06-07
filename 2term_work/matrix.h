#ifndef UNIVERSITY_MATRIX_H
#define UNIVERSITY_MATRIX_H


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdarg.h>

typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;


Matrix *createMatrix(int rows, int cols);


void makeIdentity(Matrix *mat);


void zeroMatrix(Matrix *mat);


Matrix *createZeroMatrix(int rows, int cols);


Matrix *createIdentityMatrix(int size);


Matrix *createMatrixWithElements(int rows, int cols, ...);


Matrix *createMatrixWithElementsFrom1D(int rows, int cols, double *elements);


Matrix *createMatrixWithElementsFrom2D(int rows, int cols, double **elements);


void copyEntries(Matrix *dst, const Matrix *src);


Matrix *copyMatrix(const Matrix *mat);


void destroyMatrix(Matrix *mat);


void transpose(Matrix *dst, const Matrix *mat);


int isZero(const Matrix *mat);


int isIdentity(const Matrix *mat);


int isSquare(const Matrix *m);

#endif //UNIVERSITY_MATRIX_H