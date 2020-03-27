#make clean;
#make DIM=3 SPLICE_DEPTH=5;
#./vptree ../../../input/pc/geocity.txt 200000 
#vim my_s.out

#make DIM=7 SPLICE_DEPTH=8;
#./vptree ../../../input/pc/covtype.7d 200000
#./vptree ../../../input/pc/mnist.7d 200000
#./vptree ../../../input/pc/random.7d 200000


make clean;
make DIM=3 SPLICE_DEPTH=8;
./vptree ../../../input/pc/geocity.txt 200000

make clean;
make DIM=2 SPLICE_DEPTH=4;
./vptree ../../../input/pc/geocity.txt 200000

make clean;
make DIM=2 SPLICE_DEPTH=5;
./vptree ../../../input/pc/geocity.txt 200000

make clean;
make DIM=2 SPLICE_DEPTH=6;
./vptree ../../../input/pc/geocity.txt 200000

make clean;
make DIM=2 SPLICE_DEPTH=7;
./vptree ../../../input/pc/geocity.txt 200000

make clean;
make DIM=2 SPLICE_DEPTH=8;
./vptree ../../../input/pc/geocity.txt 200000
