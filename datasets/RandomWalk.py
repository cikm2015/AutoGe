import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import json

def loadGraph(prefix):
    filename = prefix+"-G.json"
    G = json_graph.node_link_graph(json.load(open(filename)))
    G = G.to_undirected()
    return G

def random_walks(G,all = 500000):

    walks = []
    threshold = 0.8
    start = np.random.choice(G.nodes())
    while len(walks) < all:
        if np.random.random() < threshold :
            adjs = list(G[start].keys())
            if adjs :
                end = np.random.choice(adjs)
                walks.append([start,end])
                yield [start,end]
                start = end
            else:
                start = start = np.random.choice(G.nodes())
        else :
            start = np.random.choice(G.nodes())

def walks(G):
    for i in range(1000):
        nodes = G.nodes()
        np.random.shuffle(nodes)
        for each in nodes:
            start = each
            count = 0
            while count < 5:
                adjs = list(G[start].keys())
                end  = np.random.choice(adjs)
                yield [start,end]
                start = end
                count += 1
    # for each in random_walks(G):
    #     yield each
def write2file(walks,prefix):
    with open(prefix+"-walks.txt",'w') as f:
        for each in walks:
            f.write(" ".join(each))
            f.write("\n")

if __name__ == "__main__":
    prefix = "citeseer"
    G = loadGraph(prefix)
    write2file(walks(G),prefix)