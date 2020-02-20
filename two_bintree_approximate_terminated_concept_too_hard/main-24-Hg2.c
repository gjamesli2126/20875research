#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#define space_count 9
#define max_dim 2
int super_rand(int min,int max){
    return rand()%(max+1-min)+min;
}
typedef struct node{
    int data[max_dim];
    struct node *left;
    struct node *right;
}node;
void print2D(struct node *root, int space){
    if (root == NULL) return;
    space += space_count;
    print2D(root->right, space);
    printf("\n");
    for (int i = space_count; i < space; i++) printf(" ");
    printf("(%d,%d)\n", root->data[0],root->data[1]);
    print2D(root->left, space);
}
node* super_gen_seq_arr(int number){
    node *arr;
    arr=(node*)malloc(sizeof(node)*(number+1));
    for (int i = 1; i <=number ; i++) {
        arr->data[0]=i;
        arr->data[1]=100+i;
        printf("%d ",i);
    }
    printf("\n");
    return arr;
}
node* super_gen_rand_arr(int number, int min,int max){
    node *arr;
    arr=(node*)malloc(sizeof(node)*(number+1));
    for (int i = 1; i <=number ; i++) {
        arr->data[0]=super_rand(min,max);
        arr->data[1]=super_rand(min,max);
    }
    return arr;
}
double distance_sqr(node* point1,node* point2,int dim){
    int dist=0;
    int tmp;
    for(;dim>=0;dim--){
        tmp=(point1->data[dim]-point2->data[dim]);
        dist+=tmp*tmp;
    }
    return dist;
}
void swap(node* x,node* y){
    int tmp[max_dim];
    int size= sizeof(tmp);
    memcpy(tmp,x->data,size);//x->tmp
    memcpy(x->data,y->data,size);//y->x
    memcpy(y->data,tmp,size);//tmp->y
}
node* median_by_qs(node* head,node* tail,int index){
    if(head>=tail) return NULL;//meaningless if continue
    if(head==tail-1) return head;//smaller one
    node* point;
    node* store;
    node* mid_data;
    int pivot;

    mid_data=(tail-head)/2+tail;
    while(1){
        pivot=mid_data->data[index];
        swap(mid_data,tail-1);
        store=head;
        for(point=head;point<tail;point++){//why just ++? used us index?
            if(point->data[index]<pivot){
                if(point!=store) swap(point,store);
                store++;
            }
        }
        swap(store,tail-1);
        if(store->data[index]==mid_data->data[index]) return mid_data;
        if(store>mid_data){
            tail=store;
        }
        else{
            head=store;
        }

    }
}
node* createtree(node* start,int length,int dim_take_turns, int dim){
    node* new_node;
    if(length==0) return NULL;
    new_node=median_by_qs(start,start+length,dim_take_turns);
    if(new_node!=NULL){
        dim_take_turns++;
        dim_take_turns%=dim;
        new_node->left=createtree(start,new_node-start,dim_take_turns,dim);
        new_node->right=createtree(new_node+1,start-(new_node+1)+length,dim_take_turns,dim);
    }
    return new_node;
}

void nearest()