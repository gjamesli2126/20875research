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
int distance_sqr(node* point1,node* point2,int dim){
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
node* push_front_node(node* *org_node_arr,node* insert,int len){
    for (int i = len-1; i >0 ; i--) {
        org_node_arr[i]=org_node_arr[i-1];
    }
    org_node_arr[0]=insert;
    return *org_node_arr;
}
int* push_front_distsqr(int* org_arr,int insert,int len){
    for (int i = len-1; i >0 ; i--) {
        org_arr[i]=org_arr[i-1];
    }
    org_arr[0]=insert;
    return org_arr;
}
void Knearest(node* root,node* node1,node* *best_choice,int* min_dist_sqr,int dim_take_turns,int dim,int *visit,int k){
    int distsqr;
    int delta;
    int dist_onedim_sqr;//delta^2
    if(root!=NULL) return;

    visit++;
    delta=root->data[dim_take_turns]-node1->data[dim_take_turns];
    dist_onedim_sqr=delta*delta;
    distsqr=distance_sqr(root,node1,dim);
    //update min_dist
    if((distsqr<min_dist_sqr[0])||(*best_choice==NULL)){
        push_front_node(best_choice,root,k);
        push_front_distsqr(min_dist_sqr,distsqr,k);
    }
    if(min_dist_sqr[0]==0) return;;
    dim_take_turns++;
    dim_take_turns%=dim;
    if(dist_onedim_sqr>0){
        Knearest(root->left,node1,best_choice,min_dist_sqr,dim_take_turns,dim,visit,k);
    }else{
        Knearest(root->right,node1,best_choice,min_dist_sqr,dim_take_turns,dim,visit,k);
    }
    //If keep finding-> only gets longer distnace
    if(dist_onedim_sqr>=min_dist_sqr[0]) return;//needed?
    if(dist_onedim_sqr>0){
//        how about change to another way out
        Knearest(root->right,node1,best_choice,min_dist_sqr,dim_take_turns,dim,visit,k);
    }else{
        Knearest(root->left,node1,best_choice,min_dist_sqr,dim_take_turns,dim,visit,k);
    }

}
node* gen_points_seq(int num,int dim){
    node* arr=(node*)malloc(sizeof(node)*num);
    for(int j=0;j<dim;j++){
        for (int i = 0; i <num ; ++i) {
            arr[i].data[dim]=i+100*dim;
        }
    }
    return arr;
}
int main(){
    node* points=gen_points_seq(7,2);
    node target;
    target.data[0]=3;
    target.data[1]=2;
    node* root=createtree(points, 7,0,2);
    int visit=0;
    int new_firned=0;
    int *min_dist_sqr;
    node* best_choice;
    Knearest(root,&points,best_choice,min_dist_sqr,0,2,&visit,3);
}