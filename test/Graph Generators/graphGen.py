from networkx.generators.random_graphs import erdos_renyi_graph
import random

fileName = "genGraphs\graph"
numOfGraphs = 20
maxN = 3000
minN = 3000


for i in range(numOfGraphs):
    # file = open(fileName + str(i) + ".txt", "a")
    file = open("genGraphs/GraphInput.txt", "w")
    g = erdos_renyi_graph(random.randint(minN, maxN), 0.4)
    file.write(str(len(g.nodes())) + " " + str(len(g.edges())) + "\n")
    for j in g.edges():
        file.write(str(j[0]) + " " + str(j[1]) + "\n")
    file.write("\n")

file.close()
