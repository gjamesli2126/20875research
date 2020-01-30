#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
typedef struct twoD{
    int x;
    int y;
}twoD;
typedef struct node{
    twoD data;
    struct node *left;
    struct node *right;
}node;

node* create_complete_tree(int level,int seeds){
//    static int start=1;
    if(level==0) return NULL;
    node* new_node=(node*)malloc(sizeof(node));
    srand(seeds);
    new_node->data.x=rand()%48;
    new_node->data.y=rand()%48;

//    if (start==1) root=new_node;

//    start*=0;
    printf("data: (%d,%d)\t",new_node->data.x,new_node->data.y);

    new_node->left=create_complete_tree(level-1,new_node->data.x);
    new_node->right=create_complete_tree(level-1,new_node->data.y);
    return new_node;
}
void print_bt(struct node* root,int level_count){
    if(root==NULL) return;
    usleep(0.5*1000000);
    printf("level_count %d [%d,%d]\n",level_count,root->data.x,root->data.y);
    print_bt(root->left,level_count--);
    print_bt(root->right,level_count--);

}


int main() {
    printf("start!\n");
    node* root;
    root=create_complete_tree(3,0);
    printf("\n--------------------------------------\n");
    print_bt(root,3);
    return 0;
}

