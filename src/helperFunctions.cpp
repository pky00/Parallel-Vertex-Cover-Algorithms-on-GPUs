#include "helperFunctions.h"

long long int squareSequential(int num)
{
    return num * num;
}

bool binarySearchSequential(unsigned int *arr, unsigned int l, unsigned int r, unsigned int x)
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

int *deleteVertex(CSRGraph &graph, unsigned int vertex, int *vertexDegrees, unsigned int *numDeletedVertices)
{
    if (vertexDegrees[vertex] < 0)
    {
        return vertexDegrees;
    }

    for (unsigned int i = graph.srcPtr[vertex]; i < graph.srcPtr[vertex] + graph.degree[vertex]; ++i)
    {
        unsigned int neighbor = graph.dst[i];

        if (vertexDegrees[neighbor] != -1)
        {
            --vertexDegrees[neighbor];
        }
    }

    vertexDegrees[vertex] = -1;
    ++(*numDeletedVertices);
    return vertexDegrees;
}

bool leafReductionRule(CSRGraph graph, int *vertexDegrees, unsigned int *numDeletedVertices)
{
    bool hasDeleted = false;
    bool hasChanged;
    do
    {
        hasChanged = false;
        for (unsigned int i = 0; i < graph.vertexNum; ++i)
        {
            if (vertexDegrees[i] == 1)
            {
                hasChanged = true;
                for (unsigned int j = graph.srcPtr[i]; j < graph.srcPtr[i] + graph.degree[i]; ++j)
                {
                    if (vertexDegrees[graph.dst[j]] != -1)
                    {
                        hasDeleted = true;
                        unsigned int neighbor = graph.dst[j];
                        deleteVertex(graph, neighbor, vertexDegrees, numDeletedVertices);
                    }
                }
            }
        }
    } while (hasChanged);

    return hasDeleted;
}

bool highDegreeReductionRule(CSRGraph graph, int *vertexDegrees, unsigned int *numDeletedVertices, int minimum)
{
    bool hasDeleted = false;
    bool hasChanged;
    do
    {
        hasChanged = false;
        for (unsigned int i = 0; i < graph.vertexNum; ++i)
        {
            if (vertexDegrees[i] > 0 && vertexDegrees[i] + *numDeletedVertices >= minimum)
            {
                hasChanged = true;
                hasDeleted = true;
                deleteVertex(graph, i, vertexDegrees, numDeletedVertices);
            }
        }
    } while (hasChanged);

    return hasDeleted;
}

bool triangleReductionRule(CSRGraph graph, int *vertexDegrees, unsigned int *numDeletedVertices)
{
    bool hasDeleted = false;
    bool hasChanged;
    do
    {
        hasChanged = false;
        for (unsigned int i = 0; i < graph.vertexNum; ++i)
        {
            if (vertexDegrees[i] == 2)
            {

                unsigned int neighbor1, neighbor2;
                bool foundNeighbor1 = false, keepNeighbors = false;
                for (unsigned int edge = graph.srcPtr[i]; edge < graph.srcPtr[i] + graph.degree[i]; ++edge)
                {
                    unsigned int neighbor = graph.dst[edge];
                    int neighborDegree = vertexDegrees[neighbor];
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
                    bool found = binarySearchSequential(graph.dst, graph.srcPtr[neighbor2], graph.srcPtr[neighbor2 + 1] - 1, neighbor1);

                    if (found)
                    {
                        hasChanged = true;
                        hasDeleted = true;
                        // Triangle Found
                        deleteVertex(graph, neighbor1, vertexDegrees, numDeletedVertices);
                        deleteVertex(graph, neighbor2, vertexDegrees, numDeletedVertices);
                        break;
                    }
                }
            }
        }
    } while (hasChanged);

    return hasDeleted;
}