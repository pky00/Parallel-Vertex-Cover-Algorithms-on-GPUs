#ifndef BWDWORKLIST_H
#define BWDWORKLIST_H

#include "config.h"
typedef unsigned int Ticket;
typedef unsigned long long int HT;

typedef union {
	struct {int numWaiting; int numEnqueued;};
	unsigned long long int combined;
} Counter;

struct WorkList{
	unsigned int size;
	unsigned int threshold;
    volatile int* list;
    volatile unsigned int* listNumDeletedVertices;
    volatile Ticket *tickets;
    HT *head_tail;
	int* count;
	Counter * counter;
};

__device__ bool checkThreshold(WorkList workList){

    __shared__ int numEnqueued;
    if (threadIdx.x == 0){
        numEnqueued = atomicOr(&workList.counter->numEnqueued,0);
    }
    __syncthreads();

    if ( numEnqueued >= workList.threshold){
        return false;
    } else {
        return true;
    }

}


#if __CUDA_ARCH__ < 700
	__device__ __forceinline__ void backoff()
	{
		__threadfence();
	}

	__device__ __forceinline__ void sleepBWD(unsigned int exp)
	{
		__threadfence();
	}
#else
	__device__ __forceinline__ void backoff()
	{
		__threadfence();
	}

	__device__ __forceinline__ void sleepBWD(unsigned int exp)
	{
		__nanosleep(1<<exp);
	}
#endif

__device__ unsigned int* head(HT* head_tail){
	return reinterpret_cast<unsigned int*>(head_tail) + 1;
}

__device__ unsigned int* tail(HT* head_tail) {
	return reinterpret_cast<unsigned int*>(head_tail);
}

__device__ void waitForTicket(const unsigned int P, const Ticket number, WorkList workList) {
	while (workList.tickets[P] != number)
	{
		backoff();
	}
}

__device__ bool ensureDequeue(WorkList workList){
	int Num = atomicOr(workList.count,0);
	bool ensurance = false;
	while (!ensurance && Num > 0) {
		if (atomicSub(workList.count, 1) > 0) {
			ensurance = true;
		}
		else {
			Num = atomicAdd(workList.count, 1) + 1;
		}
	}

	return ensurance;
}


__device__ bool ensureEnqueue(WorkList workList){
	int Num = atomicOr(workList.count,0);
	bool ensurance = false;
	while (!ensurance && Num < (int)workList.size)
	{
		if (atomicAdd(workList.count, 1) < (int)workList.size)
		{
			ensurance = true;
		}
		else 
		{
			Num = atomicSub(workList.count, 1) - 1;
		}
	}
	
	return ensurance;
}


__device__ void readData(int* vertexDegree_s, unsigned int * vcSize, WorkList workList, unsigned int vertexNum){	
	__shared__ unsigned int P;
	unsigned int Pos;
	if (threadIdx.x==0){
	Pos = atomicAdd(head(const_cast<HT*>(workList.head_tail)), 1);
	P = Pos % workList.size;
	waitForTicket(P, 2 * (Pos / workList.size) + 1,workList);
	}
	__syncthreads();

	for(unsigned int vertex = threadIdx.x; vertex < vertexNum; vertex += blockDim.x) {
		vertexDegree_s[vertex] = workList.list[P*vertexNum + vertex];
	}

	*vcSize = workList.listNumDeletedVertices[P];

	__syncthreads();
	if (threadIdx.x==0){
	workList.tickets[P] = 2 * ((Pos + workList.size) / workList.size);
	}
}

__device__ void putData(int* vertexDegree_s, unsigned int * vcSize, WorkList workList,unsigned int vertexNum){
	__shared__ unsigned int P;
	unsigned int Pos;
	unsigned int B;
	if (threadIdx.x==0){
	Pos = atomicAdd(tail(const_cast<HT*>(workList.head_tail)), 1);
	P = Pos % workList.size;
	B = 2 * (Pos /workList.size);
	waitForTicket(P, B, workList);
	}

	__syncthreads();

	for(unsigned int i = threadIdx.x; i < vertexNum; i += blockDim.x) {
		workList.list[i + (P)*(vertexNum)] = vertexDegree_s[i];
	}

	if(threadIdx.x == 0) {
		workList.listNumDeletedVertices[P] = *vcSize;
	}
	__threadfence();
	__syncthreads();
	if (threadIdx.x==0){
		workList.tickets[P] = B + 1;
		atomicAdd(&workList.counter->numEnqueued,1);
	}
}

__device__ inline bool enqueue(int* vertexDegree_s, WorkList workList, unsigned int vertexNum,unsigned int * vcSize){
	__shared__  bool writeData;
	if (threadIdx.x==0){
		writeData = ensureEnqueue(workList);
	}

	__syncthreads();
	
	if (writeData)
	{
		putData(vertexDegree_s, vcSize, workList, vertexNum);
	}
	
	return writeData;
}


__device__ inline bool dequeue(int* vertexDegree_s, WorkList workList, unsigned int vertexNum,unsigned int * vcSize){	
	unsigned int expoBackOff = 0;

	__shared__  bool isWorkDone;
	if (threadIdx.x==0){
		isWorkDone = false;
		atomicAdd(&workList.counter->numWaiting,1);
	}
	__syncthreads();

	__shared__  bool hasData;
	while (!isWorkDone) {

		if (threadIdx.x==0){
			hasData = ensureDequeue(workList);
		}
		__syncthreads();

		if (hasData){
			readData(vertexDegree_s, vcSize, workList, vertexNum);
			if (threadIdx.x==0){
				Counter tempCounter;
				tempCounter.numWaiting = -1;
				tempCounter.numEnqueued = -2;
				atomicAdd(&workList.counter->combined,tempCounter.combined);
			}
			return true;
		}

		if (threadIdx.x==0){
			Counter tempCounter;
			tempCounter.combined = atomicOr(&workList.counter->combined,0);
			if (tempCounter.numWaiting==gridDim.x && tempCounter.numEnqueued==0){
				isWorkDone=true;
			}
		}

		__syncthreads();
		sleepBWD(expoBackOff++);
	}
	return false;
}

__device__ inline bool dequeueParameterized(int* vertexDegree_s, WorkList workList, unsigned int vertexNum,unsigned int * vcSize, unsigned int * kFound){	
	unsigned int expoBackOff = 0;
	__shared__  bool isWorkDone;
	if (threadIdx.x==0){
		isWorkDone = false;
		atomicAdd(&workList.counter->numWaiting,1);
	}
	__syncthreads();


	__shared__  bool hasData;

	while (!isWorkDone) {

		if (threadIdx.x==0){
			hasData = ensureDequeue(workList);
		}
		__syncthreads();

		if (hasData){
			readData(vertexDegree_s, vcSize, workList, vertexNum);
			if (threadIdx.x==0){
				Counter tempCounter;
				tempCounter.numWaiting = -1;
				tempCounter.numEnqueued = -2;
				atomicAdd(&workList.counter->combined,tempCounter.combined);
			}
			return true;
		}

		if (threadIdx.x==0){
			Counter tempCounter;
			tempCounter.combined = atomicOr(&workList.counter->combined,0);
			if ((tempCounter.numWaiting==gridDim.x && tempCounter.numEnqueued==0) || atomicOr(kFound,0)){
				isWorkDone=true;
			}
		}

		__syncthreads();
		sleepBWD(expoBackOff++);
	}
	return false;
}


WorkList allocateWorkList(CSRGraph graph, Config config, unsigned int numBlocks){
	WorkList workList;
	workList.size = config.globalListSize;
	workList.threshold = config.globalListThreshold * workList.size;

	volatile int* list_d;
	volatile unsigned int * listNumDeletedVertices_d;
	volatile Ticket *tickets_d;
	HT *head_tail_d;
	int* count_d;
	Counter * counter_d;
	cudaMalloc((void**) &list_d, (graph.vertexNum) * sizeof(int) * workList.size);
	cudaMalloc((void**) &listNumDeletedVertices_d, sizeof(unsigned int) * workList.size);
	cudaMalloc((void**) &tickets_d, sizeof(Ticket) * workList.size);
	cudaMalloc((void**) &head_tail_d, sizeof(HT));
	cudaMalloc((void**) &count_d, sizeof(int));
	cudaMalloc((void**) &counter_d, sizeof(Counter));
	
	workList.list = list_d;
	workList.listNumDeletedVertices = listNumDeletedVertices_d;
	workList.tickets = tickets_d;
	workList.head_tail = head_tail_d;
	workList.count=count_d;
	workList.counter = counter_d;

	HT head_tail = 0x0ULL;
	Counter counter;
	counter.combined = 0;
	cudaMemcpy(head_tail_d,&head_tail,sizeof(HT),cudaMemcpyHostToDevice);
	cudaMemcpy((void*)list_d, graph.degree, (graph.vertexNum) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset((void*)&listNumDeletedVertices_d[0], 0, sizeof(unsigned int));
	cudaMemset((void*)&tickets_d[0], 0, workList.size * sizeof(Ticket));
	cudaMemset(count_d, 0, sizeof(int));
	cudaMemcpy(counter_d, &counter ,sizeof(Counter),cudaMemcpyHostToDevice);

	return workList;
}

void cudaFreeWorkList(WorkList workList){
	cudaFree((void*)workList.list);
	cudaFree(workList.head_tail);
	cudaFree((void*)workList.listNumDeletedVertices);
	cudaFree(workList.counter);
	cudaFree(workList.count);
}

#endif
