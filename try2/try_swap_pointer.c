#include<stdio.h>
#define COUNT 9
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
void swap(twoD *a,twoD *b){
    twoD temp;
    temp=*a;
    *a=*b;
    *b=temp;
}
int* test_recursive_ef_array_with_pointer(int* arr,int size,int now){
    if (now>size) return arr;
    arr[now]=now+62;
    //print arr
    for (int i = 0; i < size; ++i) printf("%d ",arr[i]); printf("now %d\n",now);
    now++;
    test_recursive_ef_array_with_pointer(arr,size,now);
    return arr;
}
int main(){
	twoD d1,d2;
	d1.x=1;
	d1.y=2;
	d2.x=3;
	d2.y=4;
	swap(&d1,&d2);
	printf("%d,%d\t\t%d,%d",d1.x,d1.y,d2.x,d2.y);

	twoD arr1[10],arr2[10];
    for (int i = 0; i <10 ; i++) {
        arr1[i].x=100+i;
        arr1[i].y=100+10+i;
        arr2[i].x=200+i;
        arr2[i].y=200+10+i;
    }
    //swap
    for (int i = 0; i <10 ; i++) {
        swap(&arr1[i],&arr2[i]);
    }
    //print
    printf("\n");
    for (int j = 0; j <10 ; j++) {
        printf("arr1 %d\t%d\t\tarr2 %d\t%d\n",arr1[j].x,arr1[j].y,arr2[j].x,arr2[j].y);
    }
    twoD arr3[10];
    for (int k = 0; k <10 ; ++k) {
        arr3[k]=arr1[k];
    }
    printf("\n");
    arr3[2].x=99999;
    for (int j = 0; j <10 ; j++) {
        printf("arr1 %d\t%d\t\tarr2 %d\t%d\t\tarr3 %d\t%d\n",arr1[j].x,arr1[j].y,arr2[j].x,arr2[j].y,arr3[j].x,arr3[j].y);
    }
    printf("\n");
    int arr[]={6,5,4,3,2,1,7,8,6,9,0};
    test_recursive_ef_array_with_pointer(arr,11,0);

	return 0;
}
