#include "auxFunctions.h"

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <cstdlib>
#include <assert.h>
#include <time.h>
#include <cstring>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

int comp(const void *elem1, const void *elem2)
{
	int f = *((int *)elem1);
	int s = *((int *)elem2);
	if (f > s)
		return 1;
	if (f < s)
		return -1;
	return 0;
}

bool auxBinarySearch(unsigned int *arr, unsigned int l, unsigned int r, unsigned int x)
{
	while (l <= r)
	{
		unsigned int m = l + (r - l) / 2;

		if (arr[m] == x)
			return true;

		if (arr[m] < x)
			l = m + 1;

		else
			r = m - 1;
	}

	return false;
}

bool leafReductionRule(CSRGraph &graph, unsigned int &minimum)
{
	bool hasChanged;
	do
	{
		hasChanged = false;
		for (unsigned int i = 0; i < graph.vertexNum; ++i)
		{
			if (graph.degree[i] == 1)
			{
				hasChanged = true;
				for (unsigned int j = graph.srcPtr[i]; j < graph.srcPtr[i + 1]; ++j)
				{
					if (graph.degree[graph.dst[j]] != -1)
					{
						unsigned int neighbor = graph.dst[j];
						graph.deleteVertex(neighbor);
						++minimum;
					}
				}
			}
		}
	} while (hasChanged);

	return hasChanged;
}

bool triangleReductionRule(CSRGraph &graph, unsigned int &minimum)
{
	bool hasChanged;
	do
	{
		hasChanged = false;
		for (unsigned int i = 0; i < graph.vertexNum; ++i)
		{
			if (graph.degree[i] == 2)
			{

				unsigned int neighbor1, neighbor2;
				bool foundNeighbor1 = false, keepNeighbors = false;
				for (unsigned int edge = graph.srcPtr[i]; edge < graph.srcPtr[i + 1]; ++edge)
				{
					unsigned int neighbor = graph.dst[edge];
					int neighborDegree = graph.degree[neighbor];
					if (neighborDegree > 0)
					{
						if (neighborDegree == 1 || neighborDegree == 2 && neighbor < i)
						{
							keepNeighbors = true;
							break;
						}
						else if (!foundNeighbor1)
						{
							foundNeighbor1 = true;
							neighbor1 = neighbor;
						}
						else
						{
							neighbor2 = neighbor;
							break;
						}
					}
				}

				if (!keepNeighbors)
				{
					bool found = auxBinarySearch(graph.dst, graph.srcPtr[neighbor2], graph.srcPtr[neighbor2 + 1] - 1, neighbor1);

					if (found)
					{
						hasChanged = true;
						// Triangle Found
						graph.deleteVertex(neighbor1);
						graph.deleteVertex(neighbor2);
						minimum += 2;
						break;
					}
				}
			}
		}
	} while (hasChanged);

	return hasChanged;
}

CSRGraph createCSRGraphFromFile(const char *filename)
{

	CSRGraph graph;
	unsigned int vertexNum;
	unsigned int edgeNum;

	FILE *fp;
	fp = fopen(filename, "r");

	fscanf(fp, "%u%u", &vertexNum, &edgeNum);

	graph.create(vertexNum, edgeNum);

	unsigned int **edgeList = (unsigned int **)malloc(sizeof(unsigned int *) * 2);
	edgeList[0] = (unsigned int *)malloc(sizeof(unsigned int) * edgeNum);
	edgeList[1] = (unsigned int *)malloc(sizeof(unsigned int) * edgeNum);

	for (unsigned int i = 0; i < edgeNum; i++)
	{
		unsigned int v0, v1;
		fscanf(fp, "%u%u", &v0, &v1);
		edgeList[0][i] = v0;
		edgeList[1][i] = v1;
	}

	fclose(fp);

	// Gets the degrees of vertices
	for (unsigned int i = 0; i < edgeNum; i++)
	{
		assert(edgeList[0][i] < vertexNum);
		graph.degree[edgeList[0][i]]++;
		if (edgeList[1][i] >= vertexNum)
		{
			printf("\n%d\n", edgeList[1][i]);
		}
		assert(edgeList[1][i] < vertexNum);
		graph.degree[edgeList[1][i]]++;
	}
	// Fill srcPtration array
	unsigned int nextIndex = 0;
	unsigned int *srcPtr2 = (unsigned int *)malloc(sizeof(unsigned int) * vertexNum);
	for (int i = 0; i < vertexNum; i++)
	{
		graph.srcPtr[i] = nextIndex;
		srcPtr2[i] = nextIndex;
		nextIndex += graph.degree[i];
	}
	graph.srcPtr[vertexNum] = edgeNum * 2;
	// fill Graph Array
	for (unsigned int i = 0; i < edgeNum; i++)
	{
		assert(edgeList[0][i] < vertexNum);
		assert(srcPtr2[edgeList[0][i]] < 2 * edgeNum);
		graph.dst[srcPtr2[edgeList[0][i]]] = edgeList[1][i];
		srcPtr2[edgeList[0][i]]++;
		assert(edgeList[1][i] < vertexNum);
		assert(srcPtr2[edgeList[1][i]] < 2 * edgeNum);
		graph.dst[srcPtr2[edgeList[1][i]]] = edgeList[0][i];
		srcPtr2[edgeList[1][i]]++;
	}

	free(srcPtr2);
	free(edgeList[0]);
	edgeList[0] = NULL;
	free(edgeList[1]);
	edgeList[1] = NULL;
	free(edgeList);
	edgeList = NULL;

	for (unsigned int vertex = 0; vertex < graph.vertexNum; ++vertex)
	{
		qsort(&graph.dst[graph.srcPtr[vertex]], graph.degree[vertex], sizeof(int), comp);
	}

	return graph;
}

unsigned int RemoveMaxApproximateMVC(CSRGraph graph)
{

	CSRGraph approxGraph;
	approxGraph.copy(graph);

	unsigned int minimum = 0;
	bool hasEdges = true;
	while (hasEdges)
	{
		bool leafHasChanged = false, triangleHasChanged = false;
		unsigned int iterationCounter = 0;

		do
		{
			leafHasChanged = leafReductionRule(approxGraph, minimum);
			if (iterationCounter == 0 || leafHasChanged)
			{
				triangleHasChanged = triangleReductionRule(approxGraph, minimum);
			}
			++iterationCounter;
		} while (triangleHasChanged);

		unsigned int maxV;
		int maxD = 0;
		for (unsigned int i = 0; i < approxGraph.vertexNum; i++)
		{
			if (approxGraph.degree[i] > maxD)
			{
				maxV = i;
				maxD = approxGraph.degree[i];
			}
		}
		if (maxD == 0)
			hasEdges = false;
		else
		{
			approxGraph.deleteVertex(maxV);
			++minimum;
		}
	}

	approxGraph.del();

	return minimum;
}

unsigned int getRandom(int lower, int upper)
{
	srand(time(0));
	unsigned int num = (rand() % (upper - lower + 1)) + lower;
	return num;
}

unsigned int RemoveEdgeApproximateMVC(CSRGraph graph)
{

	CSRGraph approxGraph;
	approxGraph.copy(graph);

	unsigned int minimum = 0;
	unsigned int numRemainingEdges = approxGraph.edgeNum;

	for (unsigned int vertex = 0; vertex < approxGraph.vertexNum && numRemainingEdges > 0; vertex++)
	{
		bool leafHasChanged = false, triangleHasChanged = false;
		unsigned int iterationCounter = 0;

		do
		{
			leafHasChanged = leafReductionRule(approxGraph, minimum);
			if (iterationCounter == 0 || leafHasChanged)
			{
				triangleHasChanged = triangleReductionRule(approxGraph, minimum);
			}
			++iterationCounter;
		} while (triangleHasChanged);

		if (approxGraph.degree[vertex] > 0)
		{

			unsigned int randomEdge = getRandom(approxGraph.srcPtr[vertex], approxGraph.srcPtr[vertex] + approxGraph.degree[vertex] - 1);

			numRemainingEdges -= approxGraph.degree[vertex];
			numRemainingEdges -= approxGraph.degree[approxGraph.dst[randomEdge]];
			++numRemainingEdges;
			approxGraph.deleteVertex(vertex);
			approxGraph.deleteVertex(approxGraph.dst[randomEdge]);
			minimum += 2;
		}
	}

	approxGraph.del();

	return minimum;
}

bool check_graph(CSRGraph graph)
{

	unsigned int **adj_matrix = (unsigned int **)malloc(sizeof(unsigned int *) * graph.vertexNum);

	for (unsigned int i = 0; i < graph.vertexNum; ++i)
	{
		adj_matrix[i] = (unsigned int *)malloc(sizeof(unsigned int) * graph.vertexNum);

		for (unsigned int j = 0; j < graph.vertexNum; ++j)
		{
			adj_matrix[i][j] = 0;
		}
	}

	for (unsigned int vertex = 0; vertex < graph.vertexNum; ++vertex)
	{
		for (unsigned int edge = graph.srcPtr[vertex]; edge < graph.srcPtr[vertex + 1]; ++edge)
		{
			unsigned int neighbor = graph.dst[edge];
			if (adj_matrix[vertex][neighbor] == 0)
			{
				adj_matrix[vertex][neighbor] = 1;
			}
			else
			{
				return false;
			}
		}
	}

	for (unsigned int vertex = 0; vertex < graph.vertexNum; ++vertex)
	{
		if (adj_matrix[vertex][vertex] != 0)
		{
			return false;
		}
	}

	for (unsigned int i = 0; i < graph.vertexNum; ++i)
	{
		free(adj_matrix[i]);
	}
	free(adj_matrix);

	return true;
}

void performChecks(CSRGraph graph, Config config)
{
	assert(check_graph(graph) == true);
	assert(ceil(log2((float)config.blockDim)) == floor(log2((float)config.blockDim)));
	assert(ceil(log2((float)config.globalListSize)) == floor(log2((float)config.globalListSize)));
}

void setBlockDimAndUseGlobalMemory(Config &config, CSRGraph graph, int maxSharedMemPerSM, long long maxGlobalMemory, int maxNumThreadsPerSM,
								   int maxThreadsPerBlock, int maxThreadsPerMultiProcessor, int numOfMultiProcessors, int minimum)
{
	long long minNumBlocks = (maxNumThreadsPerSM / maxThreadsPerBlock) * numOfMultiProcessors;
	if (config.numBlocks)
	{
		minNumBlocks = (long long)config.numBlocks;
	}

	long long minStackSize;
	if (config.blockDim)
	{
		long long NumBlocks = (maxNumThreadsPerSM / config.blockDim) * numOfMultiProcessors;
		minStackSize = ((long long)minimum * (long long)(graph.vertexNum + 1) * (long long)sizeof(int)) * NumBlocks;
	}
	else
	{
		minStackSize = ((long long)minimum * (long long)(graph.vertexNum + 1) * (long long)sizeof(int)) * minNumBlocks;
	}

	int numSharedMemVariables = 50;
	long long globalListSize;
	if (config.version == HYBRID)
	{
		globalListSize = (long long)config.globalListSize * (long long)(graph.vertexNum + 1) * (long long)sizeof(int);
	}
	else
	{
		globalListSize = 0;
	}
	long long consumedGlobalMem = (long long)(1024 * 1024 * 1024 * 2.5) + globalListSize;
	long long availableGlobalMem = maxGlobalMemory - consumedGlobalMem;
	long long maxNumBlocksGlobalMem = MIN(availableGlobalMem / ((long long)minimum * (long long)(graph.vertexNum + 1) * (long long)sizeof(int)), maxNumThreadsPerSM * numOfMultiProcessors / 64);
	long long minNumBlocksGlobalMem = MIN(availableGlobalMem / ((long long)minimum * (long long)(graph.vertexNum + 1) * (long long)sizeof(int)), maxNumThreadsPerSM * numOfMultiProcessors / maxThreadsPerBlock);
	long long minBlockDimGlobalMem = maxNumThreadsPerSM * numOfMultiProcessors / maxNumBlocksGlobalMem;
	minBlockDimGlobalMem = pow(2, floor(log2((double)minBlockDimGlobalMem)));
	if ((long long)(consumedGlobalMem + minStackSize) > maxGlobalMemory && maxNumBlocksGlobalMem < 1)
	{
		fprintf(stderr, "\nPlease Choose A WorkList Size smaller than : %d ", config.globalListSize);
		exit(0);
	}
	else if ((long long)(consumedGlobalMem + minStackSize) > maxGlobalMemory)
	{
		config.numBlocks = maxNumBlocksGlobalMem;
	}

	long long minBlockDim = MIN(1024, minBlockDimGlobalMem);
	for (long long i = 64; i < 1024; i = i * 2)
	{
		if (maxNumThreadsPerSM * numOfMultiProcessors / i <= maxNumBlocksGlobalMem)
		{
			minBlockDim = i;
			break;
		}
	}

	long long maxBlockDim = 1024;
	long long optimalBlockDim = maxBlockDim;
	bool useSharedMem = false;

	for (long long blockDim = minBlockDim; blockDim <= maxBlockDim; blockDim *= 2)
	{
		long long maxBlocksPerSMBlockDim = maxNumThreadsPerSM / blockDim;
		long long sharedMemNeeded = (graph.vertexNum + MAX(graph.vertexNum, 2 * blockDim) + numSharedMemVariables) * sizeof(int);
		long long sharedMemPerSM = maxBlocksPerSMBlockDim * sharedMemNeeded;

		if (maxSharedMemPerSM >= sharedMemPerSM)
		{
			optimalBlockDim = blockDim;
			useSharedMem = true;
			break;
		}
	}

	printf("\nOptimal BlockDim : %d\n", optimalBlockDim);
	fflush(stdout);
	if (config.blockDim == 0)
	{
		config.blockDim = optimalBlockDim;
	}
	else
	{
		if (config.blockDim < minBlockDim)
		{
			fprintf(stderr, "\nPlease Choose A BlockDim greater than or equal to : %d\n", minBlockDim);
			exit(0);
		}
		else if (config.blockDim < optimalBlockDim && useSharedMem == 1)
		{
			useSharedMem = 0;
			if (config.userDefMemory && (config.useGlobalMemory == 0))
			{
				fprintf(stderr, "\nCannot use shared memory with this configuration, please choose a greater blockDim.\n", minBlockDim);
				exit(0);
			}
			printf("\nTo use shared memory choose a greater blockDim.\n");
		}
	}
	printf("\nUse Shared Mem : %d\n", useSharedMem);

	if (!config.userDefMemory)
	{
		config.useGlobalMemory = !useSharedMem;
	}
}

void printResults(Config config, unsigned int maxApprox, unsigned int edgeApprox, double timeMax, double timeEdge, unsigned int minimum, float timeMin, unsigned int numblocks,
				  unsigned int numBlocksPerSM, int numThreadsPerSM, unsigned int numVertices, unsigned int numEdges, unsigned int k_found)
{

	char outputFilename[500];
	strcpy(outputFilename, "Results/Results.csv");

	FILE *output_file = fopen(outputFilename, "a");

	fprintf(
		output_file,
		"%s,%s,%u,%u,%s,%s,%d,%f,%d,%d,%d,%u,%d,%d,%u,%f,%u,%f,%u,%u,%u,%f\n",
		config.outputFilePrefix, config.graphFileName, numVertices, numEdges, asString(config.instance), asString(config.version), config.globalListSize, config.globalListThreshold,
		config.startingDepth, config.useGlobalMemory, config.blockDim, numBlocksPerSM, numThreadsPerSM, config.numBlocks, maxApprox, timeMax, edgeApprox, timeEdge, minimum, config.k, k_found, timeMin);

	fclose(output_file);
}

void printResults(Config config, unsigned int maxApprox, unsigned int edgeApprox, double timeMax, double timeEdge, unsigned int minimum, float timeMin,
				  unsigned int numVertices, unsigned int numEdges, unsigned int k_found)
{

	char outputFilename[500];
	strcpy(outputFilename, "Results/Results.csv");

	FILE *output_file = fopen(outputFilename, "a");

	fprintf(
		output_file,
		"%s,%s,%u,%u,%s,%s,  ,  ,  ,  ,  ,  ,  ,  ,%u,%f,%u,%f,%u,%u,%u,%f\n",
		config.outputFilePrefix, config.graphFileName, numVertices, numEdges, asString(config.instance), asString(config.version), maxApprox, timeMax,
		edgeApprox, timeEdge, minimum, config.k, k_found, timeMin * 1000);

	fclose(output_file);
}