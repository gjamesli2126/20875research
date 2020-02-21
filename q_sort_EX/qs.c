#include<stdio.h>
#include<stdlib.h>
void quicksort(int number[25],int first,int last){
   int i, j, pivot, temp;

   if(first<last){
      pivot=first;
      i=first;
      j=last;

      while(i<j){//if cross mid-> STOP
         while(number[i]<=number[pivot]&&i<last) i++;
            
         while(number[j]>number[pivot]) j--;
         if(i<j){
            temp=number[i];
            number[i]=number[j];
            number[j]=temp;
         }
      }

      temp=number[pivot];
      number[pivot]=number[j];
      number[j]=temp;
      quicksort(number,first,j-1);
      quicksort(number,j+1,last);

   }
}

int main(){
   int i, count, number[1024];

   printf("How many elements are u going to enter?: ");
   scanf("%d",&count);
	srand(time(NULL));
   printf("Enter %d elements: ", count);
   for(i=0;i<count;i++){
	number[i]=rand();
	printf("%d ",number[i]);
   }
   printf("\n");
//      scanf("%d",&number[i]);

   quicksort(number,0,count-1);

   printf("Order of Sorted elements: ");
   for(i=0;i<count;i++)
      printf(" %d",number[i]);

   return 0;
}
