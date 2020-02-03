

#ifndef TWO_BINTREE_APPROXIMATE_QUICKSORT_H
#define TWO_BINTREE_APPROXIMATE_QUICKSORT_H
int partition(int *arr,int starting_index,int ending_index);
void quick_sort(int *arr,int starting_index,int ending_index);
typedef struct twoD{
    int x;
    int y;
    int level;
}twoD;
typedef struct node{
    twoD data;
    struct node *left;
    struct node *right;
}node;
void bubblesort(twoD *arr,size_t arr_size,int x_or_y);
#endif //TWO_BINTREE_APPROXIMATE_QUICKSORT_H
