
#include "CSRGraphRep.h"

#include <stdio.h>
#include <stdint.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <assert.h>
using namespace std;

void CSRGraph::create(unsigned int xn,unsigned int xm){
        vertexNum =xn;
        edgeNum =xm;

        dst = (unsigned int*)malloc(sizeof(unsigned int)*2*edgeNum);
        srcPtr = (unsigned int*)malloc(sizeof(unsigned int)*(vertexNum+1));
        degree = (int*)malloc(sizeof(int)*vertexNum);
        
        for(unsigned int i=0;i<vertexNum;i++){
            degree[i]=0;
        }
    }

void CSRGraph::copy(CSRGraph g){
    vertexNum =g.vertexNum;
    edgeNum =g.edgeNum;

    dst = (unsigned int*)malloc(sizeof(unsigned int)*2*edgeNum);
    srcPtr = (unsigned int*)malloc(sizeof(unsigned int)*(vertexNum+1));
    degree = (int*)malloc(sizeof(int)*vertexNum);

    for(unsigned int i = 0;i<vertexNum;i++){
        srcPtr[i] = g.srcPtr[i];
        degree[i] = g.degree[i];
    }
    srcPtr[vertexNum] = g.srcPtr[vertexNum];

    for(unsigned int i=0;i<2*edgeNum;i++){
        dst[i] = g.dst[i];
    }

}

void CSRGraph::deleteVertex(unsigned int v){
    assert(v < vertexNum);
    assert(srcPtr[v]+degree[v] <= 2*edgeNum);
    for(int i = srcPtr[v];i<srcPtr[v]+degree[v];i++){
        int neighbor = dst[i];
        assert(neighbor < vertexNum);
        int last = srcPtr[neighbor]+degree[neighbor];
        assert(last <= 2*edgeNum);
        for(int j = srcPtr[neighbor];j<last;j++){
            if(dst[j]==v){
                dst[j] = dst[last-1];
                if(degree[neighbor]!=-1){
                    degree[neighbor]--;
                }
            }
        } 
    }
    degree[v]=-1;
}

unsigned int CSRGraph::findMaxDegree(){
    unsigned int max = 0;
    unsigned int maxd = 0;
    for(unsigned int i=0;i<vertexNum;i++){
        if(degree[i]>maxd){
            maxd = degree[i];
            max = i;
        }
    }
    return max;
}

void CSRGraph::printGraph(){
        cout<<"\nDegree Array : ";
        for(unsigned int i=0;i<vertexNum;i++){
            cout<<degree[i]<<" ";
        }

        cout<<"\nsrcPtration Array : ";
        for(unsigned int i=0;i<vertexNum;i++){
            cout<<srcPtr[i]<<" ";
        }

        cout<<"\nCGraph Array : ";
        for(unsigned int i=0;i<edgeNum*2;i++){
            cout<<dst[i]<<" ";
        }
        cout<<"\n";
    }

void CSRGraph::del(){
    free(dst);
    dst = NULL;
    free(srcPtr);
    srcPtr = NULL;
    free(degree);
    degree = NULL;
}