#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <stdbool.h>
#define COUNT 20
#define DIM 3
#define DATASET_NUM 5

typedef struct point{
    float values[DIM];
    int th;
}point;

typedef struct node{
    point data;
    struct node* left;
    struct node* right;
}node;
int mypow(int x,int y){
    int result=1;
    for (int i = 0; i <y ; ++i) {
        result*=x;
    }
    return result;
}
void print_nD_arr(point* arr){
    int size=arr[0].th;
    printf("index");
    for (int k = 0; k <DIM ; ++k) printf("\t\tdata[%d]\t",k);
    printf("\t\tth\n");

    for (int i = 0; i <=size ; ++i) {
        printf("%d\t\t",i);
        for (int j = 0; j <DIM ; ++j) {
            printf("%f\t\t",arr[i].values[j]);
        }
        printf("%d\n",arr[i].th);
    }
}
void swap(point *x,point *y){
    point tmp;
    tmp=*x;
    *x=*y;
    *y=tmp;
}
point *super_gen_seq_arr(int number,bool reversed){
    int i,dim,j;
    point *arr=(point*)malloc(sizeof(point)*(number+1));
    for (i = 1; i <=number ; i++) {
        j=i;
        if(reversed==true) {
            j=number-i+1;
        }
        for (dim = 0; dim <DIM ; dim++) {
            arr[i].values[dim]=(float)(dim*100+j);//init
        }
        arr[i].th=1;//init
    }
    arr[0].th=number;
    return arr;
}
point *super_gen_rand_arr(int number,int max){
    srand(time(NULL));
    int i,dim;
    point *arr=(point*)malloc(sizeof(point)*(number+1));
    for (i = 1; i <=number ; i++) {
        for (dim = 0; dim <DIM ; dim++) {
            arr[i].values[dim]=(float)(rand() % (max+1));//should use this

//            arr[i].values[dim]=((float)(rand()%100));//init
        }
        arr[i].th=1;//init
    }
    arr[0].th=number;
    return arr;
}
point* deep_copy(point *arr){
    int size=arr[0].th;
    point* newarr=(point*)malloc(sizeof(point)*(size+1));
    memcpy(newarr,arr, sizeof(point)*(size+1));
    return newarr;
}

int print_test_qsort(point* arr){
    int val=0;
    for (int i = 1; i <=arr[0].th ; ++i) {
        val+=mypow(10,arr[0].th-i)*(int)arr[i].values[0];
    }
    return val;
}
void quicksort(point *orgarr,int first,int last,int for_which_dim){
    int from_first,from_last,pivot;
//    int testing;
//    int test_from_first_val;
//    int test_from_last_val;
//    int test_pivot_val;
//    testing=print_test_qsort(orgarr);
    if(for_which_dim>DIM){
        printf("dim Err into quick sort\n");
        EXIT_FAILURE;
    }
    if(first<last){
        pivot=first;
        from_first=first;
        from_last=last;
        while(from_first<from_last){//if left index & right index not cross mid-> continue
            //if not normal-> move the index
            while((orgarr[from_first].values[for_which_dim]<=orgarr[pivot].values[for_which_dim])&&(from_first<last)) from_first++;
            //if not normal-> move the index
            while(orgarr[from_last].values[for_which_dim]>orgarr[pivot].values[for_which_dim]) from_last--;
            //            //if valid first and last index-> swap two chosen points (1 at right and another ar left)
            if(from_first<from_last)    swap(&orgarr[from_first],&orgarr[from_last]);
//            otherwise continue
//            printf("----\n");
//            print_nD_arr(orgarr);
//            usleep(1000000*1);
//            print_nD_arr(orgarr);
        }
        //change the pivot to the right side of the chosen point
        swap(&orgarr[pivot],&orgarr[from_last]);
        //insert node for right side of the tree
        quicksort(orgarr,first,from_last-1,for_which_dim);
        //insert node for left side of the tree
        quicksort(orgarr,from_last+1,last,for_which_dim);
    }
}
void print2DUtil(node *root, int space){
    if (root == NULL) return;
    int i;
    space += COUNT;

    print2DUtil(root->right, space);
    printf("\n");
    for (i = COUNT; i < space; i++) printf(" ");
    printf("(");
    for (i = 0; i <DIM ; i++) {
        printf("%.1f  ",root->data.values[i]);
    }
    printf(")\n");
//    printf("(%d,%d)\n", root->data.,root->data.y);
    print2DUtil(root->left, space);
}
void print_node(node* root){
    int i;

    printf("(");
    for (i = 0; i <DIM ; i++) {
        printf("%.1f ",root->data.values[i]);
    }
    printf(")th:%d\n",root->data.th);
}
int print_bt(node* root){
    static int count=0;
    int i;
    if(root==NULL) return 0;
//    usleep(0.1*1000000);
    printf("(");
    for (i = 0; i <DIM ; i++) {
        printf("%.1f ",root->data.values[i]);
    }
    printf(")th:%d\n",root->data.th);
    count++;
    print_bt(root->left);
    print_bt(root->right);
    return count;
}

point* super_selection(point *orgarr,const char *up_down,int choose_dim,int split_portion){
    int portion=100/split_portion;// for annoy should change here! maybe: int->float
    int orgsorted_size=orgarr[0].th;
    point *new_arr;
    int new_arr_size;
    int i;
    //orginial_arr_size=sorted_arr_size
    point *sorted_orgarr=deep_copy(orgarr);
    quicksort(sorted_orgarr,1,orgsorted_size,choose_dim);
    int mid_index=(1+orgsorted_size)/portion;
//    printf("mid_INdex:\t%d\n",mid_index);
    if(orgsorted_size==1){
        new_arr=(point*)malloc(sizeof(point));
        new_arr[0].th=0;
        for(i=0;i<DIM;i++) new_arr[0].values[i]=sorted_orgarr[1].values[i];//deleted one or previous one
        return new_arr;
    }
    if(strcmp(up_down,"down")==0){
//        printf("DOWN\n");
        new_arr_size=mid_index;
        new_arr=(point*)malloc(sizeof(point)*(1+new_arr_size));
        for(i=1;i<=new_arr_size;i++) new_arr[i]=sorted_orgarr[i];
        for(i=0;i<DIM;i++) new_arr[0].values[i]=((sorted_orgarr[mid_index].values[i]+sorted_orgarr[mid_index+1].values[i])/2);//deleted one or previous one
    }else if(strcmp(up_down,"up")==0){
//        printf("UP\n");
        new_arr_size=orgsorted_size-mid_index;// for annoy should change here!
        new_arr=(point*)malloc(sizeof(point)*(1+new_arr_size));
        for(i=1;i<=new_arr_size;i++) new_arr[i]=sorted_orgarr[mid_index+i];
        for(i=0;i<DIM;i++) new_arr[0].values[i]=((sorted_orgarr[mid_index].values[i]+sorted_orgarr[mid_index+1].values[i])/2);//deleted one or previous one
    }else{
        printf("Debug: arr is empty & super_selection failed!!!\n");
        exit(0);
    }

    new_arr[0].th=new_arr_size;
    return new_arr;
}
node* convert_2_KDtree_code(point* arr,int th,int brute_force_range,int chosen_dim,int split_portion){
    node* new_node=(node*)malloc(sizeof(node));
    point* arr_left;//=(point*) malloc(sizeof(point)*(arr[0].th+1));
    point* arr_right;//=(point*) malloc(sizeof(point)*(arr[0].th+1));
    int i;
//    printf("\nEach recusrsion array\n");
//    print_nD_arr(arr);
    chosen_dim++;
    chosen_dim%=DIM;
    printf("Current Dim %d\n",chosen_dim);
    arr_left=(super_selection(arr,"down",chosen_dim,split_portion));
    arr_right=(super_selection(arr,"up",chosen_dim,split_portion));
    new_node->data.th=th;
    if(arr_left[0].th>=brute_force_range){
        for(i=0;i<DIM;i++) new_node->data.values[i]= arr_left[0].values[i];
        printf("L\n");
        print_nD_arr(arr_left);
        print_node(new_node);
        new_node->left=convert_2_KDtree_code(arr_left,th++,brute_force_range,chosen_dim,split_portion);
    }else{
        for(i=0;i<DIM;i++) new_node->data.values[i]= arr_left[0].values[i];
        printf("L----NULL\n");
        print_nD_arr(arr_left);
        print_node(new_node);
        new_node->left=NULL;
    }
    if(arr_right[0].th>=brute_force_range){
        for(i=0;i<DIM;i++) new_node->data.values[i]= arr_right[0].values[i];
        printf("R\n");
        print_nD_arr(arr_right);
        print_node(new_node);
        new_node->right=convert_2_KDtree_code(arr_right,th++,brute_force_range,chosen_dim,split_portion);
    }else{
        for(i=0;i<DIM;i++) new_node->data.values[i]= arr_right[0].values[i];
        printf("R----NULL\n");
        print_nD_arr(arr_right);
        print_node(new_node);
        new_node->right=NULL;
    }
    printf("------------------pop------------------------\n");
    return new_node;
}

node* convert_2_KDtree(point* arr, int split_portion){
    return convert_2_KDtree_code(arr,1,1,-1,split_portion);
}
void push_front(point* org_arr,point desire_push,int k){
//    printf("----------------------------------------------------\n");
    //need to update the arr[0].th as well!
    int i;
    org_arr[0].th+=1-(int)(k<=org_arr[0].th);
//    printf("%d\n",org_arr[0].th);
    for (i = org_arr[0].th; i>1 ; i--) {
//        printf(" %d",i);
        org_arr[i]=org_arr[i-1];
    }
//    printf("\n");
    org_arr[1]=desire_push;
//    return org_arr;
}

void k_nearest_search_code(int k,node* root,bool approximate,point target,int chosen_dim,point* nearest_points){
    //under occasion: approximate==true && only one point
    // this recursion is for approximate kNN search where k=1
    if(root==NULL) return;
    else printf("--->%.1f",root->data.values[chosen_dim]);
    if(nearest_points[1].values[chosen_dim]!=root->data.values[chosen_dim] || nearest_points[0].th==0){
        printf("S\t");//S means store!
        push_front(nearest_points,root->data,k);
    }//need modified when k>1
    if(target.values[chosen_dim]<root->data.values[chosen_dim]){
        chosen_dim++;
        chosen_dim%=DIM;
        k_nearest_search_code(k,root->left,approximate,target,chosen_dim,nearest_points);
    }else{
        chosen_dim++;
        chosen_dim%=DIM;
        k_nearest_search_code(k,root->right,approximate,target,chosen_dim,nearest_points);
    }


}
point* k_nearest_search(int k,node* tree,bool approximate,point target){
    point* nearest_points=(point*)malloc(sizeof(point)*(k+1));
    nearest_points[0].th=0;
    k_nearest_search_code(k,tree,approximate,target,0,nearest_points);
    printf("\n");
    return nearest_points;
}
int main(){
//    point* orgarr;
//    orgarr=deep_copy(super_gen_seq_arr(DATASET_NUM));
//    orgarr=deep_copy(super_gen_rand_arr(DATASET_NUM));
//    print_nD_arr(orgarr);


    point* orgarr;
//    orgarr=super_gen_seq_arr(DATASET_NUM,true);
    orgarr=super_gen_rand_arr(DATASET_NUM,80);
    print_nD_arr(orgarr);//print!
//    point* arr2;
/* test deepcopy--successful
 * arr2=orgarr;//link
    arr2=deep_copy(orgarr);//deep copy
    arr2[0].values[0]=99999;
    print_nD_arr(orgarr);
 */
/*
//    test swap & quick sort
//    point* testarr=super_gen_seq_arr(7,true);
    point* testarr=super_gen_rand_arr(21);
//    testarr[0].values[0]=99999;testarr[0].values[1]=99999;
    print_nD_arr(testarr);
//    swap(&testarr[3],&testarr[6]);
    quicksort(testarr,1,21,2);
    printf("End\n");
    print_nD_arr(testarr);
*/
/*
    //test super_selection
    printf("\n------------------------------------------------------------------\n");
    point* qsarr=deep_copy(orgarr);quicksort(qsarr,1,DATASET_NUM,0);print_nD_arr(qsarr);
    qsarr=deep_copy(orgarr);quicksort(qsarr,1,DATASET_NUM,1);print_nD_arr(qsarr);
    qsarr=deep_copy(orgarr);quicksort(qsarr,1,DATASET_NUM,2);print_nD_arr(qsarr);

    printf("\n------------------------------------------------------------------\n");
    print_nD_arr(super_selection(orgarr,"down",0,50));//print_nD_arr(selected);
    print_nD_arr(super_selection(orgarr,"down",1,50));//print_nD_arr(selected);
    print_nD_arr(super_selection(orgarr,"down",2,50));//print_nD_arr(selected);
    printf("\n------------------------------------------------------------------\n");
    print_nD_arr(super_selection(orgarr,"up",0,50));//print_nD_arr(selected);
    print_nD_arr(super_selection(orgarr,"up",1,50));//print_nD_arr(selected);
    print_nD_arr(super_selection(orgarr,"up",2,50));//print_nD_arr(selected);
*/
/*
    //test push
    printf("------------test push\n");
    point target={{51,32,61},0};
    point target1={{1,32,61},0};
    point* org=malloc(sizeof(point)*4);

    push_front(org,target,3);print_nD_arr(org);
    push_front(org,target,3);print_nD_arr(org);
    push_front(org,target,3);print_nD_arr(org);
    push_front(org,target1,3);print_nD_arr(org);
    push_front(org,target1,3);print_nD_arr(org);
*/



//  test buliding KD tree //bug fixed//succeed
    node *tree;
    tree=convert_2_KDtree(orgarr,50);//only code for 50, not yet solved other portions!
    print_bt(tree);
    print2DUtil(tree,0);
/*
    //test searching k=1
    point target={{3,103,204},0};
    printf("%.1f,%.1f,%.1f\n",target.values[0],target.values[1],target.values[2]);
    point* found=k_nearest_search(1,tree,true,target);//true: approximate search
    print_nD_arr(found);
*/

/*
//test searching k>1--untest
    point target={{3,103,204},0};
    printf("%.1f,%.1f,%.1f\n",target.values[0],target.values[1],target.values[2]);
    point* found=k_nearest_search(3,tree,true,target);//true: approximate search
    print_nD_arr(found);
*/

    return 0;
}