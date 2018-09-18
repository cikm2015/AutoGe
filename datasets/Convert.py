import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import json
label_dict = {}
def loadFeats(feats_file = "citeseer/citeseer.content"):
    G = nx.DiGraph()
    with open(feats_file) as f:
        feats_lines = f.readlines()

    for line in feats_lines:
        line = line.split()
        name = line[0]
        label = line[-1]
        if label not in label_dict:
            label_dict[label] = len(label_dict)
        feats = line[1:-1]
        G.add_node(name,id=name,label=label_dict[label],feature = feats,val = False,test=False)
    return G

def loadEdges(G,filename = "citeseer/citeseer.cites"):
    with open(filename) as f:
        lines = f.readlines()

    for line in lines:
        cited, citing = line.split()
        # if cited == citing:
        #     continue
        if cited in G.node and citing in G.node:
            G.add_edge(citing, cited)


def toFeatsArray(G):
    array = []
    id_map = {}
    index_map = {}
    for index,each in enumerate(G.node):
        array.append(G.node[each]['feature'])
        id_map[each] = index
        index_map[index] = each
    return array,id_map,index_map

def toClassMap(G):
    labels = {}

    for each in G.node:
        labels[each] = G.node[each]['label']
    return labels


def createTestData(G, index_map):
    length = len(G)
    indexs = [i for i in range(length)]
    np.random.shuffle(indexs)
    testEnd = int(length*0.2)
    valEnd = int(length*0.3)
    mark(G,indexs[:valEnd],index_map,'test')
    mark(G,indexs[testEnd:valEnd],index_map,'val')
def mark(G,indexs,index_map,target):
    for index in indexs:
        G.node[index_map[index]][target] = True

def save(G,feats,id_map,class_map,prefix="citeseer"):
    G = G.to_undirected()
    G_json=json_graph.node_link_data(G)
    with open(prefix+"-G.json","w") as f:
        json.dump(G_json,f)
    feats_arr = np.array(feats)
    np.save(prefix+"-feats.npy",feats_arr)
    with open(prefix+"-id_map.json",'w') as f:
        f.write(json.dumps(id_map))
    with open(prefix+"-class_map.json",'w') as f:
        f.write(json.dumps(class_map))

def convert(prefix):
    feats_file = prefix+"/"+prefix+".content"
    edge_file =  prefix+"/"+prefix+".cites"
    G = loadFeats(feats_file)
    loadEdges(G,edge_file)
    feats, id_map, index_map = toFeatsArray(G)
    labels = toClassMap(G)
    createTestData(G, index_map)
    save(G,feats,id_map,labels,prefix=prefix)

if __name__ == "__main__":
    prefix = "citeseer"
    convert(prefix)