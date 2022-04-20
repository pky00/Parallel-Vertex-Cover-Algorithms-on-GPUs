#ifndef HELPERFUNC_H
#define HELPERFUNC_H

#include "CSRGraphRep.h"

long long int squareSequential(int num);
int *deleteVertex(CSRGraph &graph, unsigned int vertex, int *vertexDegrees, unsigned int *numDeletedVertices);
bool leafReductionRule(CSRGraph graph, int *vertexDegrees, unsigned int *numDeletedVertices);
bool highDegreeReductionRule(CSRGraph graph, int *vertexDegrees, unsigned int *numDeletedVertices, int minimum);
bool triangleReductionRule(CSRGraph graph, int *vertexDegrees, unsigned int *numDeletedVertices);

#endif