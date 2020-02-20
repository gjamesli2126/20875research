#include <stdio.h>
#include "quicksort.h"
//cannot use qsort, bubble sort only

void swap(int *a,int *b){
    *a=*a+*b;
    *b=*a-*b;
    *a=*a-*b;
}
void bubblesort(twoD *arr,size_t arr_size,char x_or_y){
    printf("size %d\n",arr_size);
    int i,j;
    for(i=0;i<arr_size-1;i++){
        for(j=0;j<arr_size-i-1;j++){
            if(arr[j]>arr[j+1]) swap(&arr[j],&arr[j+1]);
        }
    }
}
int partition(int *arr, int starting_index, int ending_index){
    int pivot=arr[ending_index];
    for (int i = starting_index; i <ending_index ; ++i) {
        if(arr[i]<pivot){
            i++;
            swap(&arr[starting_index+1],&arr[i]);
        }
    }
    swap(&arr[starting_index],&arr[ending_index]);
    return starting_index;
}
void quick_sort(int *arr,int starting_index,int ending_index){
    if(starting_index<ending_index){
        int partition_index=partition(arr,starting_index,ending_index);
        quick_sort(arr,starting_index,partition_index-1);
        quick_sort(arr,partition_index+1,ending_index);
    }
}
void print_arr(int *arr,size_t arr_size){
    for (int i = 0; i < arr_size ; ++i)    printf("%d ",arr[i]);
    printf("\n");
}

