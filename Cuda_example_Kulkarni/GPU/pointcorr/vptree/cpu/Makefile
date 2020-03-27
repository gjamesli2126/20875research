BENCHMARK = pointcorr_vptree

all: $(BENCHMARK)

include ../../../Makefile.common

BIN_DEPS = $(BENCHMARK).o $(HARNESS_OBJ)

$(BENCHMARK): $(BIN_DEPS) 
	$(CC) -o $(BENCHMARK) $(BIN_DEPS)  $(LIBS)

$(BENCHMARK).o: $(BENCHMARK).cpp
	$(CC) $(CFLAGS) -I. -c $(BENCHMARK).cpp

clean:
	rm -f *~ $(BENCHMARK) *.o *.csv
