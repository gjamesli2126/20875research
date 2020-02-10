#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#define COUNT 13
#define DIM 2
#define DATASET_NUM 7
typedef struct point{
    float values[DIM];
    int level;
}point;

void swap(point *x,point *y){
    double tmp[DIM];
    memcpy(tmp,  x->values, sizeof(tmp));
    memcpy(x->values, y->values, sizeof(tmp));
    memcpy(y->values, tmp,  sizeof(tmp));
}

int main(){
	point p1,p2;
	p1.values[0]=3;	p1.values[1]=2;
	p2.values[0]=4;	p2.values[1]=91;
	swap(&p1,&p2);
	
	printf("%f %f %f %f",p1.values[0],p1.values[1],p2.values[0],p2.values[1]);
	
}
