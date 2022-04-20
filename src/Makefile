USE_COUNTERS=0

DEFS= -DUSE_COUNTERS=$(USE_COUNTERS)

NVCC_FLAGS= -maxrregcount=31 -O3 -w -gencode arch=compute_70,code=sm_70 -std=c++11 

output: auxFunctions.o CSRGraphRep.o main.o Sequential.o helperFunctions.o SequentialParameterized.o
	nvcc $(NVCC_FLAGS) auxFunctions.o CSRGraphRep.o helperFunctions.o Sequential.o SequentialParameterized.o main.o -o output

main.o: main.cu 
	nvcc -c $(NVCC_FLAGS) $(DEFS) main.cu

auxFunctions.o: auxFunctions.cpp auxFunctions.h config.h
	g++ -c -O3 $(DEFS) auxFunctions.cpp
	
CSRGraphRep.o: CSRGraphRep.cpp CSRGraphRep.h
	g++ -c -O3 $(DEFS) CSRGraphRep.cpp

Sequential.o: Sequential.cpp Sequential.h helperFunctions.o
	g++ -c -O3 $(DEFS) Sequential.cpp

SequentialParameterized.o: SequentialParameterized.cpp SequentialParameterized.h helperFunctions.o
	g++ -c -O3 $(DEFS) SequentialParameterized.cpp

helperFunctions.o: helperFunctions.cpp helperFunctions.h
	g++ -c -O3 $(DEFS) helperFunctions.cpp

clean:
	rm -f *.o output *.csv
