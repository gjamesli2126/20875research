make clean;
make DIM=7;
./vptree -v 200000 ../../../input/pc/random.7d > ../gpu_lockstep_bitmap/std_s.out
vim ../gpu_lockstep_bitmap/std_s.out
