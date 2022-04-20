#ifndef SEQUENTIALSTACK_H
#define SEQUENTIALSTACK_H

#include <stdint.h>

struct Stack
{
    unsigned int size;
    int top;
    int *stack;
    unsigned int *stackNumDeletedVertices;
    void print(int numVertices);
};

#endif