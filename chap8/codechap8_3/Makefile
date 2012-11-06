CC = g++
LFLAGS = -O2 
RFLAGS = -larmadillo

symQR.out: symQR.cpp libMC.o
	$(CC) $(LFLAGS) symQR.cpp libMC.o -o symQR.out $(RFLAGS)

libMC.o: libMC.h libMC.cpp
	$(CC) $(LFLAGS) -c libMC.cpp $(RFLAGS)

run: symQR.out
	./symQR.out

clear: 
	rm *.o *.out
