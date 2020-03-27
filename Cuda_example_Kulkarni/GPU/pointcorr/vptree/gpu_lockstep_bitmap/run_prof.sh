#make clean;
#make DIM=7;
#./vptree -v -s 200000 ../../../input/pc/covtype.7d
#./vptree -v -s 200000 ../../../input/pc/mnist.7d 
#./vptree -v -s 200000 ../../../input/pc/random.7d 

make clean;
make DIM=3;
./vptree -s 200000 ../../../input/pc/geocity.txt
./vptree 200000 ../../../input/pc/geocity.txt

make clean;
make DIM=2;
./vptree -s 200000 ../../../input/pc/geocity.txt
./vptree 200000 ../../../input/pc/geocity.txt




#vim my.out
