#ifndef COUNTERS_H
#define COUNTERS_H

#include <string.h>
#include "config.h"

enum CounterName { ENQUEUE=0, DEQUEUE, LEAF_REDUCTION, HIGH_DEGREE_REDUCTION, TRIANGLE_REDUCTION, MAX_DEGREE, PREPARE_LEFT_CHILD, PREPARE_RIGHT_CHILD, PUSH_TO_STACK, POP_FROM_STACK, TERMINATE, MAX_DEPTH, NUM_COUNTERS };

struct Counters {
    unsigned long long tmp[NUM_COUNTERS];
    unsigned long long totalTime[NUM_COUNTERS];
};

static __device__ void initializeCounters(Counters* counters) {
    #if USE_COUNTERS
    if(threadIdx.x == 0) {
        for(unsigned int i = 0; i < NUM_COUNTERS; ++i) {
            counters->totalTime[i] = 0;
        }
    }
    #endif
}

static __device__ void maxDepth(int stackTop, Counters* counters) {
    #if USE_COUNTERS
        if(threadIdx.x == 0) {
            if(counters->totalTime[MAX_DEPTH] < stackTop){
                counters->totalTime[MAX_DEPTH] = stackTop;
            }
        }
    #endif
    }

static __device__ void startTime(CounterName counterName, Counters* counters) {
#if USE_COUNTERS
    if(threadIdx.x == 0) {
        counters->tmp[counterName] = clock64();
    }
#endif
}

static __device__ void endTime(CounterName counterName, Counters* counters) {
#if USE_COUNTERS
    if(threadIdx.x == 0) {
        counters->totalTime[counterName] += clock64() - counters->tmp[counterName];
    }
#endif
}

void printCountersInFile(Config config, Counters * counters_d, unsigned int numBlocks){

    Counters * counters;
    counters = (Counters *)malloc(sizeof(Counters)*numBlocks);
    cudaMemcpy(counters, counters_d, sizeof(Counters)*numBlocks, cudaMemcpyDeviceToHost);

    char filename[100];
    strcpy(filename,config.graphFileName);

    char * token = strtok(filename, "/");
    // loop through the string to extract all other tokens
    char * outputfile;
    while( token != NULL ) {
        outputfile = token;
        token = strtok(NULL, "/");
    }

    char outputFilename[500];
    strcpy(outputFilename,"Counters/Counters_");
    strcat(outputFilename,config.outputFilePrefix);
    strcat(outputFilename,"_");
    strcat(outputFilename,outputfile);

    FILE *output_file = fopen(outputFilename, "w"); 

    fprintf(output_file,"BLOCK_NO,ENQUEUE,DEQUEUE,LEAF_REDUCTION,HIGH_DEGREE_REDUCTION,TRIANGLE_REDUCTION,MAX_DEGREE,PREPARE_LEFT_CHILD,PREPARE_RIGHT_CHILD,PUSH_TO_STACK,POP_FROM_STACK,TERMINATE,MAX_DEPTH\n");
 
    for(unsigned int i = 0;i<numBlocks;++i){
        fprintf(output_file,"%u",i);
        for(unsigned int j = 0;j<NUM_COUNTERS;++j){
            fprintf(output_file,",%llu",counters[i].totalTime[j]);
        }
        fprintf(output_file,"\n");
    }

    free(counters);
    fclose(output_file);
}

__device__ int get_smid(void) {
    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret) );
    return ret;
}

void printNodesPerSM(Config config, int * NODES_PER_SM,  int numOfSMs){

    int * NODES_PER_SM_host;
    NODES_PER_SM_host = (int *)malloc(sizeof(int)*numOfSMs);
    cudaMemcpy(NODES_PER_SM_host, NODES_PER_SM, sizeof(int)*numOfSMs, cudaMemcpyDeviceToHost);

    char filename[100];
    strcpy(filename,config.graphFileName);

    char * token = strtok(filename, "/");
    // loop through the string to extract all other tokens
    char * outputfile;
    while( token != NULL ) {
        outputfile = token;
        token = strtok(NULL, "/");
    }

    char outputFilename[500];
    strcpy(outputFilename,"NODES_PER_SM/NODES_PER_SM_");
    strcat(outputFilename,config.outputFilePrefix);
    strcat(outputFilename,"_");
    strcat(outputFilename,outputfile);

    FILE *output_file = fopen(outputFilename, "w"); 

	for(unsigned int i = 0;i<numOfSMs;++i){
		char str[100] = "SM_";
		fprintf(output_file,"SM_%u",i+1);
		if (i < numOfSMs - 1){
			fprintf(output_file,",");
		}
	}
	fprintf(output_file,"\n");
 
    for(unsigned int i = 0;i<numOfSMs;++i){
		fprintf(output_file,"%d",NODES_PER_SM_host[i]);
		if (i < numOfSMs - 1){
			fprintf(output_file,",");
		}
    }

    free(NODES_PER_SM_host);
    fclose(output_file);
}


#endif