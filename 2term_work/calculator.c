#include "libmatrix.h"

void printMatrix(Matrix *m) {
    for (int r = 0; r < m->rows; r++) {
        for (int c = 0; c < m->cols; c++) {
            printf("%.2lf ", m->data[r][c]);
        }
        printf("\n");
    }
}

Matrix *inputMatrix() {
    int rows, cols;
    char line[100];
    printf("Rows:");
    fgets(line, 100, stdin);
    while (sscanf(line, "%d", &rows) != 1) {
        printf("Wrong input, write a number:");
        fgets(line, 100, stdin);
    };
    printf("Cols:");
    fgets(line, 100, stdin);
    while (sscanf(line, "%d", &cols) != 1) {
        printf("Wrong input, write a number:");
        fgets(line, 100, stdin);
    }
    printf("Write matrix\n");
    Matrix *m = createMatrix(rows, cols);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            while (scanf("%lf", &(m->data[r][c])) != 1) {
                printf("Wrong input, write the numbers:\n");
                fflush(stdin);
                if (r == rows - 1 && c == cols - 1) {
                    continue;
                }
                //scanf("%lf", &(m->data[r][c]));
            }
        }
    }
    getc(stdin);
    return m;
}

int main(int argc, char *argv[]) {
    printf("\n\tMatrix Calculator\n");
    printf("\nThis calculator can do:\n");
    printf("1. add - adds two matrices;\n"
           "2. inverse - finds the inverse matrix;\n"
           "3. multiply - multiplies two matrices;\n"
           "4. power - multiplies the matrix by itself n times;\n"
           "5. conjugate - UAU^-1;\n"
           "6. transpose -transposes the matrix;\n"
           "7. det - determinant of the matrix;\n"
           "8. / - right-sided matrix division;\n"
           "9. \\ -  left-sided matrix division;\n"
           "10. multiplyScalar - multiplying a matrix by a scalar;\n"
           "11. exit - ends the program.\n");
    char cmd[200];
    while (1) {
        int Flag = 0;
        printf("\n\tChoose the action\n");
        printf("\nWrite: add, inverse, multiply, multiplyScalar, power, conjugate, transpose, det, /, \\, exit\n");
        printf(">");
        fgets(cmd, sizeof(cmd), stdin);
        cmd[strlen(cmd) - 1] = 0;
        if (!strcmp(cmd, "add")) {
            printf("First matrix:\n");
            Matrix *m1 = inputMatrix();
            printf("Second matrix:\n");
            Matrix *m2 = inputMatrix();
            add(m1, m1, m2, &Flag);
            if (Flag == 0) {
                printf("Result:\n");
                printMatrix(m1);
            }
            destroyMatrix(m1);
            destroyMatrix(m2);
            Flag = 0;
        } else if (!strcmp(cmd, "inverse")) {
            printf("Write matrix:\n");
            Matrix *m = inputMatrix();
            Matrix *min = createMatrix(m->rows, m->cols), *cof = createMatrix(m->rows, m->cols);
            double det = mat_invert(m, m, min, cof);
            printf("Minors:\n");
            printMatrix(min);
            printf("Cofactors:\n");
            printMatrix(cof);
            printf("Inverse:\n");
            printMatrix(m);
            printf("Determinant: %lf\n", det);
            destroyMatrix(m);
            destroyMatrix(min);
            destroyMatrix(cof);
        } else if (!strcmp(cmd, "multiply")) {
            printf("First matrix:\n");
            Matrix *m1 = inputMatrix();
            printf("Second matrix:\n");
            Matrix *m2 = inputMatrix();
            Matrix *md = createMatrix(m1->rows, m2->cols);
            multiplyMatrix(md, m1, m2, &Flag);
            if (Flag == 0) {
                printf("Result:\n");
                printMatrix(md);
            }
            destroyMatrix(m1);
            destroyMatrix(m2);
            destroyMatrix(md);
            Flag = 0;
        } else if (!strcmp(cmd, "power")) {
            printf("Write matrix:\n");
            Matrix *m1 = inputMatrix();
            int power;
            printf("Power:");
            //getc(stdin);
            while (scanf("%d", &power) != 1) {
                printf("Wrong input, write the number:\n");
                fflush(stdin);
            }
            if (power < 2) {
                printf("Nothing to do\n");
                destroyMatrix(m1);
                continue;
            }
            Matrix *m2 = copyMatrix(m1);
            Matrix *mp = copyMatrix(m1);
            for (int i = 2; i <= power; i++) {
                multiplyMatrix(mp, m1, m2, 0);
                copyEntries(m1, mp);
            }
            printf("Result:\n");
            printMatrix(mp);
            destroyMatrix(m1);
            destroyMatrix(m2);
            destroyMatrix(mp);
        } else if (!strcmp(cmd, "conjugate")) {
            printf("Enter U: \n");
            Matrix *U = inputMatrix();
            if (!isSquare(U)) {
                printf("U must be square\n");
                destroyMatrix(U);
                continue;
            }
            printf("Enter A: \n");
            Matrix *A = inputMatrix();
            if (!isSquare(A)) {
                printf("A must be square\n");
                destroyMatrix(U);
                destroyMatrix(A);
                continue;
            }
            Matrix *res = createMatrix(U->rows, U->cols);
            multiplyMatrix(res, U, A, 0);
            mat_invert(U, U, NULL, NULL);
            copyEntries(A, res);
            multiplyMatrix(res, A, U, 0);
            printf("UAU^-1:\n");
            printMatrix(res);
            destroyMatrix(res);
            destroyMatrix(U);
            destroyMatrix(A);
            Flag = 0;
        } else if (!strcmp(cmd, "transpose")) {
            printf("Write matrix:\n");
            Matrix *M = inputMatrix();
            Matrix *MT = createMatrix(M->cols, M->rows);
            transpose(MT, M);
            printf("Result:\n");
            printMatrix(MT);
            destroyMatrix(M);
            destroyMatrix(MT);
        } else if (!strcmp(cmd, "det")) {
            printf("Write matrix:\n");
            Matrix *m = inputMatrix();
            Matrix *res = createMatrix(m->cols, m->rows);
            double det = mat_determinant(m, NULL);
            printf("det: %.2lf\n", det);
            destroyMatrix(m);
            destroyMatrix(res);
        } else if (!strcmp(cmd, "/")) {
            printf("First matrix:\n");
            Matrix *m1 = inputMatrix();
            printf("Second matrix:\n");
            Matrix *m2 = inputMatrix();
            if (!isSquare(m2)) {
                printf("Second matrix must be square\n");
                destroyMatrix(m1);
                destroyMatrix(m2);
                continue;
            }
            mat_invert(m2, m2, NULL, NULL);
            Matrix *res = createMatrix(m1->cols, m2->rows);
            multiplyMatrix(res, m1, m2, 0);
            printf("Result (AB^-1):\n");
            printMatrix(res);
            destroyMatrix(m1);
            destroyMatrix(m2);
            destroyMatrix(res);
        } else if (!strcmp(cmd, "\\")) {
            printf("First matrix:\n");
            Matrix *m1 = inputMatrix();
            if (!isSquare(m1)) {
                printf("First matrix must be square\n");
                destroyMatrix(m1);
                continue;
            }
            printf("Second matrix:\n");
            Matrix *m2 = inputMatrix();
            mat_invert(m1, m1, NULL, NULL);
            Matrix *res = createMatrix(m1->cols, m1->rows);
            multiplyMatrix(res, m1, m2, 0);
            printf("Result (A^-1B):\n");
            printMatrix(res);
            destroyMatrix(m1);
            destroyMatrix(m2);
            destroyMatrix(res);
        } else if (!strcmp(cmd, "multiplyScalar")) {
            printf("Write matrix:\n");
            Matrix *m = inputMatrix();
            printf("Scalar:\n");
            double val;
            while (scanf("%lf", &val) != 1) {
                printf("Wrong input, write the number:\n");
                fflush(stdin);
            }
            //getc(stdin);
            multiplyScalar(m, m, val);
            printf("Result:\n");
            printMatrix(m);
            destroyMatrix(m);
        } else if (!strcmp(cmd, "exit")) {
            break;
        } else {
            printf("Invalid command\n");
        }
        memset(cmd, 0, sizeof(cmd));
    }
    return 0;
}