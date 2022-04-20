from networkx.generators.random_graphs import random_regular_graph
import random
 
fileName = "genGraphs/regularGraph"
numOfGraphs = 5
maxN = 150
minN = 100

regularity = 5


for i in range(numOfGraphs):
    file = open(fileName+str(i)+".txt","a")
    retry = True
    while retry:
        try:
            g = random_regular_graph(regularity,random.randint(minN,maxN))
            retry = False
        except:
            retry = True
    file.write(str(len(g.nodes()))+" "+str(len(g.edges()))+"\n")
    for j in g.edges():
        file.write(str(j[0])+" "+str(j[1])+"\n")
    file.write("\n")