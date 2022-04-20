
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include "config.h"
#include "stack.cuh"
#include "Counters.cuh"
#include "BWDWorkList.cuh"
#include "helperFunctions.cuh"

#if USE_GLOBAL_MEMORY
__global__ void GlobalWorkListParameterized_global_kernel(Stacks stacks, WorkList workList, CSRGraph graph, Counters* counters
    , int* first_to_dequeue_global, int* global_memory, unsigned int * k, unsigned int * kFound, int* NODES_PER_SM) {
#else
__global__ void GlobalWorkListParameterized_shared_kernel(Stacks stacks, WorkList workList, CSRGraph graph, Counters* counters
    , int* first_to_dequeue_global, unsigned int * k, unsigned int * kFound, int* NODES_PER_SM) {
#endif

    __shared__ Counters blockCounters;
    initializeCounters(&blockCounters);

    #if USE_COUNTERS
        __shared__ unsigned int sm_id;
        if (threadIdx.x==0){
            sm_id=get_smid();
        }
    #endif

    unsigned int stackSize = (stacks.minimum + 1);
    volatile int * stackVertexDegrees = &stacks.stacks[blockIdx.x * stackSize * graph.vertexNum];
    volatile unsigned int * stackNumDeletedVertices = &stacks.stacksNumDeletedVertices[blockIdx.x * stackSize];
    int stackTop = -1;

    // Define the vertexDegree_s
    unsigned int numDeletedVertices;
    unsigned int numDeletedVertices2;
    
    #if USE_GLOBAL_MEMORY
    int * vertexDegrees_s = &global_memory[graph.vertexNum*(2*blockIdx.x)];
    int * vertexDegrees_s2 = &global_memory[graph.vertexNum*(2*blockIdx.x + 1)];
    #else
    extern __shared__ int shared_mem[];
    int * vertexDegrees_s = shared_mem;
    int * vertexDegrees_s2 = &shared_mem[graph.vertexNum];
    #endif

    bool dequeueOrPopNextItr = true; 
    __syncthreads();

    __shared__ bool first_to_dequeue;
    if (threadIdx.x==0){
        if(atomicCAS(first_to_dequeue_global,0,1) == 0) { 
            first_to_dequeue = true;
        } else {
            first_to_dequeue = false;
        }
    }
    __syncthreads();
    if (first_to_dequeue){
        for(unsigned int vertex = threadIdx.x; vertex < graph.vertexNum; vertex += blockDim.x) {
            vertexDegrees_s[vertex]=workList.list[vertex];
        }
        numDeletedVertices = workList.listNumDeletedVertices[0];
        dequeueOrPopNextItr = false;
    }

    __syncthreads();
    __shared__ unsigned int kIsFound;

    while(true){
        if(threadIdx.x==0){
            kIsFound = atomicOr(kFound,0);
        }
        __syncthreads();
        if(kIsFound){
            break;
        }
        
        if(dequeueOrPopNextItr) {
            if(stackTop != -1) { // Local stack is not empty, pop from the local stack
                startTime(POP_FROM_STACK,&blockCounters);
                popStack(graph.vertexNum, vertexDegrees_s, &numDeletedVertices, stackVertexDegrees, stackNumDeletedVertices, &stackTop);
                endTime(POP_FROM_STACK,&blockCounters);

                #if USE_COUNTERS
                    if (threadIdx.x==0){
                        atomicAdd(&NODES_PER_SM[sm_id],1);
                    }
                #endif
                __syncthreads();
            } else { // Local stack is empty, read from the global workList
                startTime(TERMINATE,&blockCounters);
                startTime(DEQUEUE,&blockCounters);
                if(!dequeueParameterized(vertexDegrees_s, workList, graph.vertexNum, &numDeletedVertices,kFound)) {   
                    endTime(TERMINATE,&blockCounters);
                    break;
                }
                endTime(DEQUEUE,&blockCounters);

                #if USE_COUNTERS
                    if (threadIdx.x==0){
                        atomicAdd(&NODES_PER_SM[sm_id],1);
                    }
                #endif
            }
        }

        __syncthreads();

        unsigned int iterationCounter = 0, numDeletedVerticesLeaf, numDeletedVerticesTriangle, numDeletedVerticesHighDegree;
        do{
            startTime(LEAF_REDUCTION,&blockCounters);
            numDeletedVerticesLeaf = leafReductionRule(graph.vertexNum, vertexDegrees_s, graph, vertexDegrees_s2);
            endTime(LEAF_REDUCTION,&blockCounters);
            numDeletedVertices += numDeletedVerticesLeaf;
            if(iterationCounter==0 || numDeletedVerticesLeaf > 0 || numDeletedVerticesHighDegree > 0){
                startTime(TRIANGLE_REDUCTION,&blockCounters);
                numDeletedVerticesTriangle = triangleReductionRule(graph.vertexNum, vertexDegrees_s, graph, vertexDegrees_s2);
                endTime(TRIANGLE_REDUCTION,&blockCounters);
                numDeletedVertices += numDeletedVerticesTriangle;
            } else {
                numDeletedVerticesTriangle = 0;
            }
            if(iterationCounter==0 || numDeletedVerticesLeaf > 0 || numDeletedVerticesTriangle > 0){
                startTime(HIGH_DEGREE_REDUCTION,&blockCounters);
                numDeletedVerticesHighDegree = highDegreeReductionRule(graph.vertexNum, vertexDegrees_s, graph, vertexDegrees_s2,numDeletedVertices,*k+1);
                endTime(HIGH_DEGREE_REDUCTION,&blockCounters);
                numDeletedVertices += numDeletedVerticesHighDegree;
            }else {
                numDeletedVerticesHighDegree = 0;
            }
        }while(numDeletedVerticesTriangle > 0 || numDeletedVerticesHighDegree > 0);

        unsigned int numOfEdges = findNumOfEdges(graph.vertexNum, vertexDegrees_s, vertexDegrees_s2);

        if(numDeletedVertices > *k || numOfEdges>=square(*k-numDeletedVertices)+1) { // Reached the bottom of the tree, no minimum vertex cover found
            dequeueOrPopNextItr = true;

        } else {
            unsigned int maxVertex;
            int maxDegree;
            startTime(MAX_DEGREE,&blockCounters);
            findMaxDegree(graph.vertexNum, &maxVertex, &maxDegree, vertexDegrees_s, vertexDegrees_s2);
            endTime(MAX_DEGREE,&blockCounters);

            __syncthreads();
            if(maxDegree == 0) { // Reached the bottom of the tree, minimum vertex cover possibly found
                
                if(threadIdx.x==0){
                    if(numDeletedVertices<=*k){
                        atomicOr(kFound,1);
                    }
                }

                dequeueOrPopNextItr = true;

            } else { // Vertex cover not found, need to branch

                startTime(PREPARE_RIGHT_CHILD,&blockCounters);
                deleteNeighborsOfMaxDegreeVertex(graph,vertexDegrees_s, &numDeletedVertices, vertexDegrees_s2, &numDeletedVertices2, maxDegree, maxVertex);
                endTime(PREPARE_RIGHT_CHILD,&blockCounters);

                __syncthreads();

                bool enqueueSuccess;
                if(checkThreshold(workList)){
                    startTime(ENQUEUE,&blockCounters);
                    enqueueSuccess = enqueue(vertexDegrees_s2, workList, graph.vertexNum, &numDeletedVertices2);
                } else  {
                    enqueueSuccess = false;
                }
                
                __syncthreads();
                
                if(!enqueueSuccess) {
                    startTime(PUSH_TO_STACK,&blockCounters);
                    pushStack(graph.vertexNum, vertexDegrees_s2, &numDeletedVertices2, stackVertexDegrees, stackNumDeletedVertices, &stackTop);
                    maxDepth(stackTop, &blockCounters);
                    endTime(PUSH_TO_STACK,&blockCounters);
                    __syncthreads(); 
                } else {
                    endTime(ENQUEUE,&blockCounters);
                }

                startTime(PREPARE_LEFT_CHILD,&blockCounters);
                // Prepare the child that removes the neighbors of the max vertex to be processed on the next iteration
                deleteMaxDegreeVertex(graph, vertexDegrees_s, &numDeletedVertices, maxVertex);
                endTime(PREPARE_LEFT_CHILD,&blockCounters);

                dequeueOrPopNextItr = false;
            }
        }
        __syncthreads();
    }

    #if USE_COUNTERS
    if(threadIdx.x == 0) {
        counters[blockIdx.x] = blockCounters;
    }
    #endif

}
