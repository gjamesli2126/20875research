#include <stdio.h>
#include <stdlib.h>
#include <time.h>
typedef struct twoD{
    int x;
    int y;
}twoD;
typedef struct node{
    twoD data;
    struct node *left;
    struct node *right;
}node;

node* create_complete_tree(int level,int seeds,node* root){
    static int start=1;
    if(level==0) return NULL;
    node* new_node=(node*)malloc(sizeof(node));
    srand(seeds);
    new_node->data.x=rand()%48;
    new_node->data.y=rand()%48;
//    printf("level: %d\t",level);
    if (start==1) root=new_node;

    start*=0;
    printf("data: (%d,%d)\t",new_node->data.x,new_node->data.y);

    new_node->left=create_complete_tree(level-1,new_node->data.x,root);
    new_node->right=create_complete_tree(level-1,new_node->data.y,root);
    return root;
}
void print_bin_tree(struct node* root,int level_count){
    if(root!=NULL){
        printf("%d\t",root->data);
//        print_bin_tree(root,level_count--);
    }
    else if(root->left!=NULL){
        print_bin_tree(root->left,level_count--);
    } else if (root->right!=NULL){
        print_bin_tree(root->right,level_count--);
    }

}


int main() {
    printf("start!\n");
    node* root;
    root=create_complete_tree(3,0,0);
    printf("\n--------------------\n");
    print_bin_tree(root,3);
    return 0;
}

