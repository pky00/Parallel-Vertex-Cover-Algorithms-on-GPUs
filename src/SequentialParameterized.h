#ifndef SEQPARAM_H
#define SEQPARAM_H

#include "CSRGraphRep.h"
#include "helperFunctions.h"
#include "stack.h"
#include <stdlib.h>

unsigned int SequentialParameterized(CSRGraph graph, unsigned int minimum, unsigned int k, unsigned int *kFound);

#endif