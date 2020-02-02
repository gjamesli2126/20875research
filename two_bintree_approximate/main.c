#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
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
void single_swap(int *a,int *b){
    *a=*a+*b;
    *b=*a-*b;
    *a=*a-*b;
}
void swap(twoD *a,twoD *b){
    twoD temp;
    temp=*a;
    *a=*b;
    *b=temp;
}
int super_pow(int x,int y){
    for (int i = 0; i < y; ++i) {
        x*=x;
    }
    return x;
}
int super_rand(int min,int max){
    return rand()%(max+1-min)+min;
}
node* create_tree(int level,int seeds){
    if(level==0) return NULL;
    node* new_node=(node*)malloc(sizeof(node));
    srand(seeds);
    new_node->data.x=rand()%16;
    new_node->data.y=rand()%16;
    new_node->data.level=level;
    printf("data: (%d,%d)_lv%d\t\n",new_node->data.x,new_node->data.y,new_node->data.level);
    new_node->left=create_tree(level-1,new_node->data.x);
    new_node->right=create_tree(level-1,new_node->data.y);
    return new_node;
}
int print_bt(struct node* root){
    static int count=0;
    if(root==NULL) return 0;
    usleep(0.1*1000000);
    printf("[%d,%d]_lv%d\n",root->data.x,root->data.y,root->data.level);
    count++;
    print_bt(root->left);
    print_bt(root->right);
    return count;
}
twoD* super_gen_arr(int number, int min,int max){
    twoD arr[number];
    for (int i = 0; i <number ; ++i) {
        arr[i].x=super_rand(min,max);
        arr[i].y=super_rand(min,max);
        arr[i].level= -1;
    }
    return arr;
}
void bubblesort(twoD *arr,int arr_size,char x_or_y){
    printf("size %d\n",arr_size);
    int i,j;
    if(x_or_y=='x'){
        for(i=0;i<arr_size-1;i++){
            for(j=0;j<arr_size-i-1;j++){
                if(arr[j].x>arr[j+1].x) swap(&arr[j],&arr[j+1]);
            }
        }
    }else if (x_or_y=='y'){
        for(i=0;i<arr_size-1;i++){
            for(j=0;j<arr_size-i-1;j++){
                if(arr[j].y>arr[j+1].y) swap(&arr[j],&arr[j+1]);
            }
        }
    }

}
twoD* super_arr_copy(twoD *arr,int arr_size){
    twoD* new_arr;
    for (int i = 0; i <arr_size ; ++i) {
        new_arr[i]=arr[i];
    }
    return new_arr;
}
twoD* super_selection(twoD* org_arr,int org_arr_size, const char* mode,const char x_or_y){
    int i;
    twoD *cpy_org_arr;
    int arr_size;
    cpy_org_arr=super_arr_copy(org_arr,org_arr_size);
    bubblesort(cpy_org_arr,org_arr_size,x_or_y);
    twoD *arr;
    if(strcmp(mode, "down") == 0){
        if(org_arr_size % 2==0) arr_size=org_arr_size/2;
        else arr_size=org_arr_size/2+1;
        arr=(twoD*)malloc(sizeof(twoD)*arr_size);
        for(i=0;i<arr_size;i++){
            arr[i]=cpy_org_arr[i];
        }
    }
    else if(strcmp(mode, "up") == 0){
        arr_size=org_arr_size/2;
        arr=(twoD*)malloc(sizeof(twoD)*arr_size);
        for(i=0;i<arr_size;i++){
            arr[i]=cpy_org_arr[(org_arr_size-arr_size)+i];
        }
    } else{
        printf("Debug: arr is empty & super_selection failed!!!\n");
        exit(0);
    }

    return arr;
}
node* create_kd_tree(int min,int max,int level){
    if(level==0) return NULL;
    //calculate number range
    
    //initialize node
    node* new_node=(node*)malloc(sizeof(node));
    new_node->data.x=super_rand(min,max);
    new_node->data.y=super_rand(min,max);
    new_node->data.level=level;
    printf("data: (%d,%d)_lv%d\t\n",new_node->data.x,new_node->data.y,new_node->data.level);
    //treuly create node in tree
    new_node->left=create_kd_tree(level-1,new_node->data.x);
    new_node->right=create_kd_tree(level-1,new_node->data.y);
    return new_node;
}
void print2DUtil(struct node *root, int space){
    if (root == NULL) return;
    space += COUNT;
    print2DUtil(root->right, space);
    printf("\n");
    for (int i = COUNT; i < space; i++) printf(" ");
    printf("(%d,%d)\n", root->data.x,root->data.y);
    print2DUtil(root->left, space);
}
void find_k_neighbor(struct node* root,int k,int count){
    printf("Find K nearest neighbor\n");
}

int main() {
    printf("start!\n");
    node* root;
//    root=create_tree(3,0);
    printf("\n--------------------------------------\n");
//    printf("Print binary tree\n");
    int count;
    twoD* init_arr;
    int init_arr_size=15;

    printf("\n--------------------------------------\n");
    print2DUtil(root,0);

    printf("\n--------------------------------------\n");

    init_arr=super_gen_arr(init_arr_size,0,24);


    printf("\n--------------------------------------\n");
//    find_k_neighbor(root,4,count);
    return 0;
}

