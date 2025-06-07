#ifndef UNIVERSITY_INVERSE_H
#define UNIVERSITY_INVERSE_H

#include "matrix.h"
#include "arithmetic.h"

void mat_minors(Matrix *dst, const Matrix *mat);


void mat_cofactors(Matrix *dst, const Matrix *minors);


double mat_determinant(const Matrix *mat, const Matrix *cofactors);


double mat_invert(Matrix *dst, const Matrix *mat, Matrix *minors, Matrix *cofactors);

#endif //UNIVERSITY_INVERSE_H
