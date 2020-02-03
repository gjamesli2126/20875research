#include <stdio.h>

void swap(int *a,int *b){
    *a=*a+*b;
    *b=*a-*b;
    *a=*a-*b;
}

int partition(int *arr, int starting_index, int ending_index){
    int pivot=arr[ending_index];
    int i;
    for (i = starting_index; i <ending_index ; ++i) {
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
void print_arr(int *arr){
	int i;
    for (i = 0; i < sizeof(arr)/ sizeof(*arr) ; ++i)	printf("%d ",arr[i]);
    printf("\n%d %d",sizeof(arr),sizeof(*arr));
    printf("\n----------------\n");
}
int main(){
    int arr[] = {10, 7, 8, 9, 1, 5};
    size_t n = sizeof(arr)/sizeof(arr[0]);
    quick_sort(arr, 0, 6);
    printf("Sorted array: ");
    print_arr(arr);
    return 0;
}
