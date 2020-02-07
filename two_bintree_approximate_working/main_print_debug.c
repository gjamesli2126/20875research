#include <stdio.h>#include <stdlib.h>#include <time.h>#include <unistd.h>#include <math.h>#define COUNT 13typedef struct twoD{    int x;    int y;    int level;}twoD;typedef struct node{    twoD data;    struct node *left;    struct node *right;}node;void print_twoD_arr(twoD* arr){    int size=arr[0].level;    printf("index\t\tx\t\ty\t\tlevel\n");    for (int i = 0; i <=size ; ++i) {        printf("%d\t\t%d\t\t%d\t\t%d\n",i,arr[i].x,arr[i].y,arr[i].level);    }}void single_swap(int *a,int *b){    *a=*a+*b;    *b=*a-*b;    *a=*a-*b;}void swap(twoD *a,twoD *b){    twoD temp;    temp=*a;    *a=*b;    *b=temp;}int super_pow(int x,int y){    for (int i = 0; i < y; ++i) {        x*=x;    }    return x;}int super_rand(int min,int max){    return rand()%(max+1-min)+min;}void print2DUtil(struct node *root, int space){    if (root == NULL) return;    space += COUNT;        print2DUtil(root->right, space);        printf("\n");        for (int i = COUNT; i < space; i++) printf(" ");        printf("(%d,%d)\n", root->data.x,root->data.y);        print2DUtil(root->left, space);}node* create_tree(int level,int seeds){    if(level==0) return NULL;    node* new_node=(node*)malloc(sizeof(node));    srand(seeds);    new_node->data.x=rand()%16;    new_node->data.y=rand()%16;    new_node->data.level=level;    printf("data: (%d,%d)_lv%d\t\n",new_node->data.x,new_node->data.y,new_node->data.level);    new_node->left=create_tree(level-1,new_node->data.x);    new_node->right=create_tree(level-1,new_node->data.y);    return new_node;}int print_bt(struct node* root){    static int count=0;    if(root==NULL) return 0;//    usleep(0.1*1000000);    printf("[%d,%d]_lv%d\n",root->data.x,root->data.y,root->data.level);    count++;    print_bt(root->left);    print_bt(root->right);    return count;}twoD* super_gen_seq_arr(int number){    twoD *arr;    arr=(twoD*)malloc(sizeof(twoD)*(number+1));    for (int i = 1; i <=number ; i++) {        arr[i].x=i;        arr[i].y=100+i;        arr[i].level= 1;        printf("%d ",i);    }    printf("\n");//    arr[0].x=0;//    arr[0].y=0;    arr[0].level=number;    return arr;}twoD* super_gen_rand_arr(int number, int min,int max){    twoD *arr;    arr=(twoD*)malloc(sizeof(twoD)*(number+1));    for (int i = 1; i <=number ; i++) {        arr[i].x=super_rand(min,max);        arr[i].y=super_rand(min,max);        arr[i].level= 1;    }//    arr[0].x=0;//    arr[0].y=0;    arr[0].level=number;    return arr;}void bubblesort(twoD *arr,int x_or_y){    int arr_size=arr[0].level;    printf("Before bubble_ size %d\n",arr_size);    int i,j;    if(x_or_y==1){        for(i=1;i<=arr_size;i++){            for(j=1;j<=arr_size-i;j++){                if(arr[j].x>arr[j+1].x) swap(&arr[j],&arr[j+1]);            }        }    }else if (x_or_y==-1){        for(i=1;i<=arr_size;i++){            for(j=1;j<=arr_size-i;j++){                if(arr[j].y>arr[j+1].y) swap(&arr[j],&arr[j+1]);            }        }    }//    printf("Bubble sort result:\n");//    print_twoD_arr(arr);}twoD* super_arr_copy(twoD *arr){    int arr_size=arr[0].level;    twoD* new_arr=(twoD*)malloc(sizeof(twoD)*(arr_size+1));    for (int i=0; i <=arr_size ; ++i) {        new_arr[i]=arr[i];    }    return new_arr;}//selected from arraytwoD* super_selection(twoD* org_arr, const char* mode,int x_or_y,int brute_force_range){    int i;    int org_arr_size=org_arr[0].level;    twoD *cpy_org_arr;    int arr_size;    if(org_arr[0].level==brute_force_range){        org_arr[0].x=-52;//for the first element can print        return org_arr;    }    cpy_org_arr=super_arr_copy(org_arr);    printf("cpy arr:\n");    print_twoD_arr(cpy_org_arr);    printf("cpy END\n");    bubblesort(cpy_org_arr,x_or_y);    twoD *arr;    if(strcmp(mode, "down") == 0){        printf("DOWN\n");        if(org_arr_size % 2==0) arr_size=org_arr_size/2;        else arr_size=org_arr_size/2+1;        arr=(twoD*)malloc(sizeof(twoD)*(arr_size+1));//arr[0] store the "size" of this array!!! -> size10 will become size11 but size[0]=10        for(i=1;i<=arr_size;i++){            arr[i]=cpy_org_arr[i];        }    }    else if(strcmp(mode, "up") == 0){        printf("UP\n");        arr_size=org_arr_size/2;        arr=(twoD*)malloc(sizeof(twoD)*arr_size);        printf("IN super selection--UP org_arr_size-arr_size:%d\n",org_arr_size-arr_size);        for(i=1;i<=arr_size;i++){            arr[i]=cpy_org_arr[(arr_size)+i];        }    } else{        printf("Debug: arr is empty & super_selection failed!!!\n");        exit(0);    }    arr[0].level=arr_size;    print_twoD_arr(arr);printf("END\n\n");    return arr;}node* convert_to_kd_tree_main(twoD* arr,int level,int brute_force_range,int x_or_y){    if(arr[0].level==brute_force_range){        printf("pop\n");        return NULL;    }    printf("\n\n\n\n");    printf("============From this arraybefore LEFT===========\n");    print_twoD_arr(arr);    printf("====================================\n");    //debuging    if(arr[2].y==102){}//debuging point    node* new_node=(node*)malloc(sizeof(node));    // calculate number range--left    twoD* selected_arr_left=super_selection(arr,"down",x_or_y,brute_force_range);//    if(selected_arr_left[0].level<brute_force_range){//        printf("pop---left\n");//        return NULL;//    }    printf("\n\t--left--convert size %d\n", selected_arr_left[0].level);    printf("Print selected_arr in convert_to_kd_tree\n");    print_twoD_arr(selected_arr_left);    printf("+++++++++++++++++++++++++++++++++\n");    new_node->data.x=selected_arr_left[selected_arr_left[0].level].x;    new_node->data.y=selected_arr_left[selected_arr_left[0].level].y;    new_node->data.level=level;    printf("left: (%d,%d)_lv%d\n",new_node->data.x,new_node->data.y,new_node->data.level);    level++;    x_or_y*=-1;    new_node->left=convert_to_kd_tree_main(selected_arr_left,level,brute_force_range,x_or_y);    // calculate number range--right    printf("============From this arraybefore RIGHT===========\n");    print_twoD_arr(arr);    printf("====================================\n");    twoD* selected_arr_right=super_selection(arr,"up",x_or_y,brute_force_range);//    if(selected_arr_right[0].level<brute_force_range){//        printf("pop---right\n");//        return NULL;//    }    printf("\n\t--right--convert size %d\n", selected_arr_right[0].level);    printf("Print selected_arr in convert_to_kd_tree\n");    print_twoD_arr(selected_arr_right);    printf("+++++++++++++++++++++++++++++++++\n");    level--;    new_node->data.x=selected_arr_right[1].x;    new_node->data.y=selected_arr_right[1].y;    new_node->data.level=level;    level++;    printf("right: (%d,%d)_lv%d\n", new_node->data.x, new_node->data.y, new_node->data.level);    new_node->right=convert_to_kd_tree_main(selected_arr_right,level,brute_force_range,x_or_y);    return new_node;}node* convert_to_kd_tree(twoD* arr){    convert_to_kd_tree_main(arr,1,1,1);}void find_k_neighbor(struct node* root,int k,int count){    printf("Find K nearest neighbor\n");}int main() {    printf("start!\n");    node* root;    root=create_tree(3,0);    printf("\n-----------------create tree Fin---------------------\n");//    printf("Print binary tree\n");    int count;    int init_arr_size=15;    twoD* init_arr;    print2DUtil(root,0);    printf("\n--------------------print 2D fin------------------\n");    printf("\n--------------------------------------\n");//    init_arr=super_gen_rand_arr(init_arr_size,0,40);//come back later    init_arr=super_gen_seq_arr(init_arr_size);    print_twoD_arr(init_arr);    printf("\n-------------genr arr fin & print arr fin-------------------------\n");    root=convert_to_kd_tree(init_arr);    printf("\n--------------covert to KD tree Fin------------------------\n");    print2DUtil(root,0);    printf("\n--------------print 2D Fin------------------------\n");    print_bt(root);    return 0;}