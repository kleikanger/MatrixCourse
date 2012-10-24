CC = g++
LFLAGS = -O3 
RFLAGS = -larmadillo

qrIter.out: qrIter.cpp
	$(CC) $(LFLAGS) -o qrIter.out qrIter.cpp $(RFLAGS)

run: qrIter.out
	./qrIter.out

clear: 
	rm *.o *.out
