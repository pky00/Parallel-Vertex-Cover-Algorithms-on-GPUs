import sys
import numpy as np

separator = " "
numOfVertices = int(sys.argv[2])

file = open(sys.argv[1], "r")
edges = file.readlines()
file.close()

counter = 0
vertices = {}
for edge in edges:
    edge = edge[: len(edge) - 1]
    array_edge = edge.split(separator)
    if vertices.get(array_edge[0], "False") == "False":
        vertices[array_edge[0]] = counter
        counter = counter + 1
    if vertices.get(array_edge[1], "False") == "False":
        vertices[array_edge[1]] = counter
        counter = counter + 1

graph = np.zeros((numOfVertices, numOfVertices))
for edge in edges:
    edge = edge[: len(edge) - 1]
    edge = edge.split(separator)
    graph[vertices[edge[0]]][vertices[edge[1]]] = 1
    graph[vertices[edge[1]]][vertices[edge[0]]] = 1

numOfEdges = 0
for i in range(numOfVertices):
    for j in range(i + 1):
        if graph[i][j] and i != j:
            numOfEdges = numOfEdges + 1

file = open("formated_" + sys.argv[1], "w")
file.write(f"{numOfVertices} {numOfEdges}\n")
for i in range(numOfVertices):
    for j in range(i + 1):
        if graph[i][j] and i != j:
            file.write(f"{i} {j}\n")
