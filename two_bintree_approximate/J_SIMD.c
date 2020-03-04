#include "J_SIMD.h"
#include <stdio.h>
#include <immintrin.h>  // portable to all x86 compilers
void printVec4(__m128 vec){
    int i;
    float array[4];
    _mm_store_ps( array, vec);
    for(i=0; i<4; i++){
        printf("%f, " , array[i]);
    }
    printf("\n");

}
void try(){
    __m128 vector1 = _mm_set_ps(4.0, 3.0, 2.0, 1.0); // high element first, opposite of C array order.  Use _mm_setr_ps if you want "little endian" element order in the source.
    __m128 vector2 = _mm_set_ps(400.0, 300.0, 200.0, 100.0);

    __m128 sum = _mm_add_ps(vector1, vector2); // result = vector1 + vector 2

//    vector1 = _mm_shuffle_ps(vector1, vector1, _MM_SHUFFLE(0,1,2,3));

    printVec4(sum);

}

