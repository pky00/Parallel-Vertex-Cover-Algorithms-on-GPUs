#ifndef AUX_H
#define AUX_H

#include "CSRGraphRep.h"
#include "config.h"

CSRGraph createCSRGraphFromFile(const char *filename);
unsigned int RemoveMaxApproximateMVC(CSRGraph graph);
unsigned int RemoveEdgeApproximateMVC(CSRGraph graph);
bool check_graph(CSRGraph graph);
void performChecks(CSRGraph graph, Config config);
void setBlockDimAndUseGlobalMemory(Config &config, CSRGraph graph, int maxSharedMemPerSM, long long maxGlobalMemory, int maxNumThreadsPerSM,
                                   int maxThreadsPerBlock, int maxThreadsPerMultiProcessor, int numOfMultiProcessors, int minimum);
void printResults(Config config, unsigned int maxApprox, unsigned int edgeApprox, double timeMax, double timeEdge, unsigned int minimum, float timeMin, unsigned int numblocks,
                  unsigned int numBlocksPerSM, int numThreadsPerSM, unsigned int numVertices, unsigned int numEdges, unsigned int k_found);
void printResults(Config config, unsigned int maxApprox, unsigned int edgeApprox, double timeMax, double timeEdge, unsigned int minimum, float timeMin,
                  unsigned int numVertices, unsigned int numEdges, unsigned int k_found);
#endif
