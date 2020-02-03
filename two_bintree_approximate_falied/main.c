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
void print_twoD_arr(twoD* arr){
    int size=arr[0].level;
    printf("index\t\tx\t\ty\t\tlevel\n");
    for (int i = 0; i <size ; ++i) {
        printf("%d\t\t%d\t\t%d\t\t%d\n",i,arr[i].x,arr[i].y,arr[i].level);
    }
}
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
void print2DUtil(struct node *root, int space){
    if (root == NULL) return;
    space += COUNT;

        print2DUtil(root->right, space);
        printf("\n");
        for (int i = COUNT; i < space; i++) printf(" ");
        printf("(%d,%d)\n", root->data.x,root->data.y);
        print2DUtil(root->left, space);

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
    twoD *arr;
    arr=(twoD*)malloc(sizeof(twoD)*(number+1));
    for (int i = 1; i <=number ; ++i) {
        arr[i].x=super_rand(min,max);
        arr[i].y=super_rand(min,max);
        arr[i].level= 1;
    }
    arr[0].level=number;
    return arr;
}
void bubblesort(twoD *arr,int x_or_y){
    int arr_size=arr[0].level;
//    printf("Before bubble_ size %d\n",arr_size);
    int i,j;
    if(x_or_y==1){
        for(i=1;i<arr_size;i++){
            for(j=1;j<arr_size-i;j++){
                if(arr[j].x>arr[j+1].x) swap(&arr[j],&arr[j+1]);
            }
        }
    }else if (x_or_y==-1){
        for(i=1;i<arr_size;i++){
            for(j=1;j<arr_size-i;j++){
                if(arr[j].y>arr[j+1].y) swap(&arr[j],&arr[j+1]);
            }
        }
    }

}
twoD* super_arr_copy(twoD *arr){
    int arr_size=arr[0].level;
    twoD* new_arr=(twoD*)malloc(sizeof(twoD)*(arr_size+1));
    for (int i=0; i <=arr_size ; ++i) {
        new_arr[i]=arr[i];
    }

    return new_arr;
}
//selected from array
twoD* super_selection(twoD* org_arr, const char* mode,int x_or_y){
    int i;
    int org_arr_size=org_arr[0].level;
    twoD *cpy_org_arr;
    int arr_size;
    cpy_org_arr=super_arr_copy(org_arr);

    bubblesort(cpy_org_arr,x_or_y);
    twoD *arr;
    if(strcmp(mode, "down") == 0){
        if(org_arr_size % 2==0) arr_size=org_arr_size/2;
        else arr_size=org_arr_size/2+1;

        arr=(twoD*)malloc(sizeof(twoD)*(arr_size+1));//arr[0] store the "size" of this array!!! -> size10 will become size11 but size[0]=10
        for(i=1;i<=arr_size;i++){
            arr[i]=cpy_org_arr[i];
        }
    }
    else if(strcmp(mode, "up") == 0){
        arr_size=org_arr_size/2;
        arr=(twoD*)malloc(sizeof(twoD)*arr_size);
        for(i=1;i<=arr_size;i++){
            arr[i]=cpy_org_arr[(org_arr_size-arr_size)+i];
        }
    } else{
        printf("Debug: arr is empty & super_selection failed!!!\n");
        exit(0);
    }
    arr[0].level=arr_size;
    return arr;
}

node* convert_to_kd_tree_main(twoD* arr,int min_index,int max_index,int level,int brute_force_range,int x_or_y){
    //min_index & max_index could be useless
    //    x:1,y:-1
    if(arr[0].level<=brute_force_range) return NULL;//If brute_force_range >0-> need to solve how to store the rest of the points
    // node space initializes
    node* new_node=(node*)malloc(sizeof(node));
    // calculate number range
    twoD* selected_arr=super_selection(arr,"down",x_or_y);
    printf("convert size %d\n", selected_arr[0].level);
    printf("Print selected_arr in convert_to_kd_tree\n");
    print_twoD_arr(selected_arr);
    //----left
    new_node->data.x=selected_arr[selected_arr[0].level].x;//last element== index
    new_node->data.y=selected_arr[selected_arr[0].level].y;//last element
    new_node->data.level=level;
    level++;
    x_or_y*=-1;
    printf("data: (%d,%d)_lv%d\t\n",new_node->data.x,new_node->data.y,new_node->data.level);
//    sleep(1);
    new_node->left=convert_to_kd_tree_main(selected_arr,min_index,max_index,level,brute_force_range,x_or_y);
    new_node->right=convert_to_kd_tree_main(selected_arr,min_index,max_index,level,brute_force_range,x_or_y);
    return new_node;
}
node* convert_to_kd_tree(twoD* arr){
    convert_to_kd_tree_main(arr,0,arr[0].level-1,1,1,1);
}
void find_k_neighbor(struct node* root,int k,int count){
    printf("Find K nearest neighbor\n");
}

int main() {
    printf("start!\n");
    node* root;
    root=create_tree(3,0);
    printf("\n--------------------------------------\n");
//    printf("Print binary tree\n");
    int count;
    int init_arr_size=15;
    twoD* init_arr;

    printf("\n--------------------------------------\n");
    print2DUtil(root,0);

    printf("\n--------------------------------------\n");

    init_arr=super_gen_arr(init_arr_size,0,40);
    print_twoD_arr(init_arr);
    printf("\n--------------------------------------\n");
    root=convert_to_kd_tree(init_arr);
    print2DUtil(root,0);
    printf("\n\n");
    print_bt(root);
    return 0;
}

