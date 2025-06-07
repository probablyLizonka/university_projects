
#ifndef UNIVERSITY_ARITHMETIC_H
#define UNIVERSITY_ARITHMETIC_H

#include "matrix.h"

void add(Matrix *dst, const Matrix *m1, const Matrix *m2, int *F);


void multiplyScalar(Matrix *dst, const Matrix *mat, double scale);


void multiplyMatrix(Matrix *dst, const Matrix *m1, const Matrix *m2, int *F);


#endif //UNIVERSITY_ARITHMETIC_H
