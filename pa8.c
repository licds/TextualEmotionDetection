#include <stdio.h>
#include<stdlib.h>
#include<time.h>
#define MAX_VALUE 20
//int rand(void)
void multiply(const int dim, const int * const a, const int * const b, int * const c) {
    int c_value;
    int times = 0;
    for ( int i = 0; i < dim * dim; i++ ){
        for ( int row = 0; row < dim ; row++) {
            c_value = 0;
            for( int col = 0; col < dim; col++) {
                c_value += a[i * dim + col] * b[col* dim + row];
            }
            c[i *dim +row] = c_value;

        
    }
}
}
/*
void multiply_transpose(const int dim, const int * const a, const int * const b_t, int * const c)
void transpose(const int dim, int * const m)
void print(const int dim, const int * const m)
void init(const int dim, int * const m) {
    time_t t;
    srand((unsigned) time(&t));
    int index = 0;

    for( int i = 0; i < dim * dim  ; i++) {
        m[index++] = rand() % MAX_VALUE;
    }
    for( int i = 0; i < index ; i ++) {
        printf("%d\n",m[i]);
    }
     

}
*/
/*
struct timeval run_and_time(
void (* mult_func)(const int, const int * const, const int * const, int * const),
const int dim,
const int * const a,
const int * const b,
int * const c
)

int verify(const int dim, const int * const c1, const int * const c2)
void run_and_test(const int dim)
*/
int main() {
    int  n = 3;
    
    //init(n,index);
    int index[9] = {1, 2, 3, 4, 5, 6,7,8,9};
    int output[9];
    multiply(3, index,index,output);
    for( int i = 0; i < 9; i++) {
        printf("%d\n",output[i]);
    }

    
}
