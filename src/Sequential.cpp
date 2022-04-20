#include "Sequential.h"

unsigned int Sequential(CSRGraph graph, unsigned int minimum)
{
    Stack stack;
    stack.size = graph.vertexNum + 1;

    stack.stack = (int *)malloc(sizeof(int) * stack.size * graph.vertexNum);
    stack.stackNumDeletedVertices = (unsigned int *)malloc(sizeof(int) * stack.size);

    for (unsigned int j = 0; j < graph.vertexNum; ++j)
    {
        stack.stack[j] = graph.degree[j];
    }
    stack.stackNumDeletedVertices[0] = 0;

    stack.top = 0;
    bool popNextItr = true;
    int *vertexDegrees = (int *)malloc(sizeof(int) * graph.vertexNum);
    unsigned int numDeletedVertices;

    while (stack.top != -1)
    {

        if (popNextItr)
        {
            for (unsigned int j = 0; j < graph.vertexNum; ++j)
            {
                vertexDegrees[j] = stack.stack[stack.top * graph.vertexNum + j];
                numDeletedVertices = stack.stackNumDeletedVertices[stack.top];
            }
            --stack.top;
        }

        bool leafHasChanged = false, highDegreeHasChanged = false, triangleHasChanged = false;
        unsigned int iterationCounter = 0;

        do
        {
            leafHasChanged = leafReductionRule(graph, vertexDegrees, &numDeletedVertices);
            triangleHasChanged = triangleReductionRule(graph, vertexDegrees, &numDeletedVertices);
            highDegreeHasChanged = highDegreeReductionRule(graph, vertexDegrees, &numDeletedVertices, minimum);

        } while (triangleHasChanged || highDegreeHasChanged);

        unsigned int maxVertex = 0;
        int maxDegree = 0;
        unsigned int numEdges = 0;

        for (unsigned int i = 0; i < graph.vertexNum; ++i)
        {
            int degree = vertexDegrees[i];
            if (degree > maxDegree)
            {
                maxDegree = degree;
                maxVertex = i;
            }

            if (degree > 0)
            {
                numEdges += degree;
            }
        }

        numEdges /= 2;

        if (numDeletedVertices >= minimum || numEdges >= squareSequential(minimum - numDeletedVertices - 1) + 1)
        {
            popNextItr = true;
        }
        else
        {
            if (maxDegree == 0)
            {
                minimum = numDeletedVertices;
                popNextItr = true;
            }
            else
            {
                popNextItr = false;
                ++stack.top;

                for (unsigned int j = 0; j < graph.vertexNum; ++j)
                {
                    stack.stack[stack.top * graph.vertexNum + j] = vertexDegrees[j];
                }
                stack.stackNumDeletedVertices[stack.top] = numDeletedVertices;

                for (unsigned int i = graph.srcPtr[maxVertex]; i < graph.degree[maxVertex] + graph.srcPtr[maxVertex]; ++i)
                {
                    deleteVertex(graph, graph.dst[i], &stack.stack[stack.top * graph.vertexNum], &stack.stackNumDeletedVertices[stack.top]);
                }

                deleteVertex(graph, maxVertex, vertexDegrees, &numDeletedVertices);
            }
        }
    }

    graph.del();
    free(stack.stack);
    free(vertexDegrees);

    return minimum;
}