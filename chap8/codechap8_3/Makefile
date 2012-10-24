CC = g++
LFLAGS = -O3 
RFLAGS = -larmadillo

symQR.out: symQR.cpp
	$(CC) $(LFLAGS) -o symQR.out symQR.cpp $(RFLAGS)

run: symQR.out
	./symQR.out

clear: 
	rm *.o *.out
