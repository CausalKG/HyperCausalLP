# ##### Create a KG
import rdflib
from rdflib import URIRef
from ampligraph.evaluation import train_test_split_no_unseen
from rdflib.namespace import Namespace, NamespaceManager
from rdflib.namespace import RDF, RDFS
from collections import ChainMap
from rdflib import Dataset
from rdflib import Graph
import networkx as nx
import pandas as pd
import numpy as np
import pickle
import json
import uuid

with open('./Data/CegWithNoCycles.txt', 'r') as f:
    cegWithNoCycles = json.load(f)

with open('./Data/VerbObject.json', 'r') as f:
        verbObject = json.load(f)

file = open('./Data//valid_ceg_data_May12.p', 'rb')  
validData = pickle.load(file)
file.close()

file = open('./Data/train_ceg_data_May12.p', 'rb')  
trainData = pickle.load(file)
file.close()

data = ChainMap(trainData, validData)

# For the mediator triples, look for depth 2 or 3 for creating the dataset. 
# Consider the Markov split at Depth 2

# Read the list of CEG ids with given Depth
file = open('./Data/trainVideoIdCEGList.pkl', 'rb')  
trainCEG = pickle.load(file)
file.close()

file = open('./Data/testVideoIdCEGList.pkl', 'rb')  
testCEG = pickle.load(file)
file.close()


fileHyper=open("./Data/CausalCLEVRERHumans_HyperGraph.txt","w")
fileTriples=open("./Data/CausalCLEVRERHumans_Triples.txt","w")

clevrerHumans = Namespace("http://semantic.bosch.com/CausalClevrerHumans/v00/")
clevrerHumansData = Namespace("http://semantic.bosch.com/CausalClevrerHumans/v00/data/")
clevrer = Namespace("http://semantic.bosch.com/CLEVRER/")
causal = Namespace("http://semantic.bosch.com/causal/v00/")
scene = Namespace("http://semantic.bosch.com/scene/v02/")
ssn = Namespace("http://www.w3.org/ns/ssn/")

# # Add for the Shape, color and material property to the graph
# # Shape: Cube, Sphere, Ball, 
# # Get unique shape, color, and material
shapeSet = set()
colorSet = set()
materialSet = set()

for ids in cegWithNoCycles:
    cegdata = data[ids].get('CEG_full')
    for k in cegdata.nodes(): 
        if 'Color' in verbObject[ids][k]['Object'][0].keys():
            colorSet.add(verbObject[ids][k]['Object'][0]['Color'])
        if 'Material' in verbObject[ids][k]['Object'][0].keys():    
            materialSet.add(verbObject[ids][k]['Object'][0]['Material'])
        if 'Shape' in verbObject[ids][k]['Object'][0].keys():    
            shapeSet.add(verbObject[ids][k]['Object'][0]['Shape'])


shape = dict()
color = dict()
material = dict()

for c in colorSet:
    colorUUID = URIRef(clevrerHumansData+str(uuid.uuid4()))   
    color[c] = colorUUID
#     g.add((colorUUID, RDF.type, URIRef(clevrerHumans+"Color"))) 
#     g.add((colorUUID, RDFS.label, rdflib.term.Literal(c))) 
    fileTriples.writelines([colorUUID,",", RDF.type,",", URIRef(clevrerHumans+"Color"),".","\n"])
    fileTriples.writelines([colorUUID,",", RDFS.label,",", rdflib.term.Literal(c),".","\n"])
    fileHyper.writelines([colorUUID,",", RDF.type,",", URIRef(clevrerHumans+"Color"),".","\n"])
    fileHyper.writelines([colorUUID,",", RDFS.label,",", rdflib.term.Literal(c),".","\n"])

for s in shapeSet:
    shapeUUID = URIRef(clevrerHumansData+str(uuid.uuid4()))  
    shape[s] = shapeUUID
#     g.add((shapeUUID, RDF.type, URIRef(clevrerHumans+"Shape"))) 
#     g.add((shapeUUID, RDFS.label, rdflib.term.Literal(s))) 
    fileTriples.writelines([shapeUUID,",", RDF.type,",", URIRef(clevrerHumans+"Shape"),".","\n"])
    fileTriples.writelines([shapeUUID,",", RDFS.label,",", rdflib.term.Literal(s),".","\n"])
    fileHyper.writelines([shapeUUID,",", RDF.type,",", URIRef(clevrerHumans+"Shape"),".","\n"])
    fileHyper.writelines([shapeUUID,",", RDFS.label,",", rdflib.term.Literal(s),".","\n"])

for m in materialSet:
    materialUUID = URIRef(clevrerHumansData+str(uuid.uuid4()))   
    material[m] = materialUUID
#     g.add((materialUUID, RDF.type, URIRef(clevrerHumans+"Material")))  
#     g.add((materialUUID, RDFS.label, rdflib.term.Literal(m))) 
    fileTriples.writelines([materialUUID,",", RDF.type,",", URIRef(clevrerHumans+"Material"),".","\n"])
    fileTriples.writelines([materialUUID,",", RDFS.label,",", rdflib.term.Literal(m),".","\n"])
    fileHyper.writelines([materialUUID,",", RDF.type,",", URIRef(clevrerHumans+"Material"),".","\n"])
    fileHyper.writelines([materialUUID,",", RDFS.label,",", rdflib.term.Literal(m),".","\n"])


objectParticipant = dict()
for c in color:
    for s in shape:
        objectParticipantUUID = URIRef(clevrerHumansData+str(uuid.uuid4()))             
        objectParticipant[c+"_"+s] = objectParticipantUUID
#         g.add(((objectParticipantUUID), RDF.type, scene.Object)) 
#         g.add(((objectParticipantUUID), RDFS.label, rdflib.term.Literal(c+" "+s))) 
        fileTriples.writelines([objectParticipantUUID,",", RDF.type,",", scene.Object,".","\n"])
        fileTriples.writelines([objectParticipantUUID,",", RDFS.label,",", rdflib.term.Literal(c+" "+s),".","\n"])
        fileHyper.writelines([objectParticipantUUID,",", RDF.type,",", scene.Object,".","\n"])
        fileHyper.writelines([objectParticipantUUID,",", RDFS.label,",", rdflib.term.Literal(c+" "+s),".","\n"])

        
#         g.add(((objectParticipantUUID), ssn.hasProperty, color[c]))
#         g.add(((objectParticipantUUID), ssn.hasProperty, shape[s]))

        fileTriples.writelines([objectParticipantUUID,",", ssn.hasProperty,",", color[c],".","\n"])
        fileTriples.writelines([objectParticipantUUID,",", ssn.hasProperty,",", shape[s],".","\n"])
        fileHyper.writelines([objectParticipantUUID,",", ssn.hasProperty,",", color[c],".","\n"])
        fileHyper.writelines([objectParticipantUUID,",", ssn.hasProperty,",", shape[s],".","\n"])

for c in color:
    for s in shape:
        for m in material:
            if c not in ['silver', 'gold']:
                objectParticipantUUID = URIRef(clevrerHumansData+str(uuid.uuid4()))             
                objectParticipant[c+"_"+m+"_"+s] = objectParticipantUUID
#                 g.add(((objectParticipantUUID), RDF.type, scene.Object)) 
#                 g.add(((objectParticipantUUID), RDFS.label, rdflib.term.Literal(c+" "+m+" "+s))) 
                fileTriples.writelines([objectParticipantUUID,",", RDF.type,",", scene.Object,".","\n"])
                fileTriples.writelines([objectParticipantUUID,",", RDFS.label,",", rdflib.term.Literal(c+" "+m+" "+s),".","\n"])
                fileHyper.writelines([objectParticipantUUID,",", RDF.type,",", scene.Object,".","\n"])
                fileHyper.writelines([objectParticipantUUID,",", RDFS.label,",", rdflib.term.Literal(c+" "+m+" "+s),".","\n"])

#                 g.add(((objectParticipantUUID), ssn.hasProperty, color[c]))
#                 g.add(((objectParticipantUUID), ssn.hasProperty, shape[s]))
#                 g.add(((objectParticipantUUID), ssn.hasProperty, material[m]))
                
                fileTriples.writelines([objectParticipantUUID,",", ssn.hasProperty,",", color[c],".","\n"])
                fileTriples.writelines([objectParticipantUUID,",", ssn.hasProperty,",", shape[s],".","\n"])
                fileTriples.writelines([objectParticipantUUID,",", ssn.hasProperty,",", material[m],".","\n"])
                fileHyper.writelines([objectParticipantUUID,",", ssn.hasProperty,",", color[c],".","\n"])
                fileHyper.writelines([objectParticipantUUID,",", ssn.hasProperty,",", shape[s],".","\n"])
                fileHyper.writelines([objectParticipantUUID,",", ssn.hasProperty,",", material[m],".","\n"])

for ids in cegWithNoCycles:
# for ids in ['00053']:
# IDS =['10016', '10038', '10063']
# for ids in IDS:
#     g.add((URIRef(clevrerHumansData+(ids)), RDF.type, scene.Scene))
    fileTriples.writelines([URIRef(clevrerHumansData+(ids)),",", RDF.type,",", scene.Scene,".","\n"])
    fileHyper.writelines([URIRef(clevrerHumansData+(ids)),",", RDF.type,",", scene.Scene,".","\n"])

    
    ceg_data = data[ids]
    ceg_full=ceg_data.get('CEG_full')
    
    threshold = 1
    # filter out all edges above threshold and grab id's
    long_edges = list(filter(lambda e: e[2] == threshold, (e for e in ceg_full.edges.data('weight'))))
    le_ids = list(e[:2] for e in long_edges)

    # remove filtered edges from graph G
    ceg_full.remove_edges_from(le_ids)

    long_edges = list(filter(lambda e: e[2] == threshold, (e for e in ceg_full.edges.data('width'))))
    le_ids = list(e[:2] for e in long_edges)
    # remove filtered edges from graph G
    ceg_full.remove_edges_from(le_ids)
    
    nodeDict = dict()
    for n in ceg_full.nodes():
        nodeDict[n] = str(uuid.uuid4())
        
    for k in ceg_full.adj.keys(): 
        cegSubjectNode = URIRef(str(k)) 
        subjectUUID = URIRef(str(nodeDict[k]))             
        for nodes in ceg_full.adj[k]:
            if ceg_full.get_edge_data(k, nodes)['weight'] > 1:
                cegObjectNode =  URIRef(str(nodes)) #URIRef(str(uuid.uuid4()))
                objectUUID =  URIRef(str(nodeDict[nodes]))
                
#                 g.add((URIRef(clevrerHumansData+(ids)), scene.includes, URIRef(clevrerHumansData+subjectUUID))) 
#                 g.add((URIRef(clevrerHumansData+subjectUUID), RDFS.label, rdflib.term.Literal(cegSubjectNode))) 
                fileTriples.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", URIRef(clevrerHumansData+subjectUUID),".","\n"])
                fileTriples.writelines([URIRef(clevrerHumansData+subjectUUID),",", RDFS.label,",", rdflib.term.Literal(cegSubjectNode),".","\n"])
                fileHyper.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", URIRef(clevrerHumansData+subjectUUID),".","\n"])
                fileHyper.writelines([URIRef(clevrerHumansData+subjectUUID),",", RDFS.label,",", rdflib.term.Literal(cegSubjectNode),".","\n"])

#                 g.add((URIRef(clevrerHumansData+(ids)), scene.includes, URIRef(clevrerHumansData+objectUUID))) 
#                 g.add((URIRef(clevrerHumansData+objectUUID), RDFS.label, rdflib.term.Literal(cegObjectNode))) 
                fileTriples.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", URIRef(clevrerHumansData+objectUUID),".","\n"])
                fileTriples.writelines([URIRef(clevrerHumansData+objectUUID),",", RDFS.label,",", rdflib.term.Literal(cegObjectNode),".","\n"])
                fileHyper.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", URIRef(clevrerHumansData+objectUUID),".","\n"])
                fileHyper.writelines([URIRef(clevrerHumansData+objectUUID),",", RDFS.label,",", rdflib.term.Literal(cegObjectNode),".","\n"])
               
                for objects in verbObject[ids][k]['Object']:
                    if objects:
                        if 'Material' in objects.keys():
#                             print(ids,k,objects, objects.keys(),objects['Material'])
#                             print(objects['Color'], objects['Material'], objects['Shape'])
                            uid = objectParticipant[objects['Color']+"_"+objects['Material']+"_"+objects['Shape']]
                            
#                             g.add((URIRef(clevrerHumansData+(ids)), scene.includes, uid))
                            fileTriples.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", uid,".","\n"])
                            fileHyper.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", uid,".","\n"])
                            
#                             g.add((URIRef(clevrerHumansData+subjectUUID), scene.hasParticipant, uid))
#                             g.add((uid, scene.isParticipantIn, URIRef(clevrerHumansData+subjectUUID)))
                            fileTriples.writelines([URIRef(clevrerHumansData+subjectUUID),",", scene.hasParticipant,",", uid,".","\n"])
                            fileTriples.writelines([uid,",", scene.isParticipantIn,",", URIRef(clevrerHumansData+subjectUUID),".","\n"])
                            fileHyper.writelines([URIRef(clevrerHumansData+subjectUUID),",", scene.hasParticipant,",", uid,".","\n"])
                            fileHyper.writelines([uid,",", scene.isParticipantIn,",", URIRef(clevrerHumansData+subjectUUID),".","\n"])
  
                        else:
                            uid = objectParticipant[objects['Color']+"_"+objects['Shape']]
                            
#                             g.add((URIRef(clevrerHumansData+(ids)), scene.includes, uid))
                            fileTriples.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", uid,".","\n"])
                            fileHyper.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", uid,".","\n"])
            
#                             g.add((URIRef(clevrerHumansData+subjectUUID), scene.hasParticipant, uid))
#                             g.add((uid, scene.isParticipantIn, URIRef(clevrerHumansData+subjectUUID)))
                            fileTriples.writelines([URIRef(clevrerHumansData+subjectUUID),",", scene.hasParticipant,",", uid,".","\n"])
                            fileTriples.writelines([uid,",", scene.isParticipantIn,",", URIRef(clevrerHumansData+subjectUUID),".","\n"])
                            fileHyper.writelines([URIRef(clevrerHumansData+subjectUUID),",", scene.hasParticipant,",", uid,".","\n"])
                            fileHyper.writelines([uid,",", scene.isParticipantIn,",", URIRef(clevrerHumansData+subjectUUID),".","\n"])
                    
                                
                    
                for objects in verbObject[ids][nodes]['Object']:
                    if objects:
                        if 'Material' in objects.keys():
                            uid = objectParticipant[objects['Color']+"_"+objects['Material']+"_"+objects['Shape']]
                            
#                             g.add((URIRef(clevrerHumansData+(ids)), scene.includes, uid))
                            fileTriples.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", uid,".","\n"])
                            fileHyper.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", uid,".","\n"])
 
#                             g.add((URIRef(clevrerHumansData+objectUUID), scene.hasParticipant, uid))
#                             g.add((uid, scene.isParticipantIn, URIRef(clevrerHumansData+objectUUID)))
                            fileTriples.writelines([URIRef(clevrerHumansData+objectUUID),",", scene.hasParticipant,",", uid,".","\n"])
                            fileTriples.writelines([uid,",", scene.isParticipantIn,",", URIRef(clevrerHumansData+objectUUID),".","\n"])
                            fileHyper.writelines([URIRef(clevrerHumansData+objectUUID),",", scene.hasParticipant,",", uid,".","\n"])
                            fileHyper.writelines([uid,",", scene.isParticipantIn,",", URIRef(clevrerHumansData+objectUUID),".","\n"])
                
                        else:
                            uid = objectParticipant[objects['Color']+"_"+objects['Shape']]
                            
#                             g.add((URIRef(clevrerHumansData+(ids)), scene.includes, uid))
                            fileTriples.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", uid,".","\n"])
                            fileHyper.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", uid,".","\n"])
                              
#                             g.add((URIRef(clevrerHumansData+objectUUID), scene.hasParticipant, uid))
#                             g.add((uid, scene.isParticipantIn, URIRef(clevrerHumansData+objectUUID)))
                            fileTriples.writelines([URIRef(clevrerHumansData+objectUUID),",", scene.hasParticipant,",", uid,".","\n"])
                            fileTriples.writelines([uid,",", scene.isParticipantIn,",", URIRef(clevrerHumansData+objectUUID),".","\n"])
                            fileHyper.writelines([URIRef(clevrerHumansData+objectUUID),",", scene.hasParticipant,",", uid,".","\n"])
                            fileHyper.writelines([uid,",", scene.isParticipantIn,",", URIRef(clevrerHumansData+objectUUID),".","\n"])

                
                if verbObject[ids][nodes]['Verbs'][0] == 'come':
                    verbObject[ids][nodes]['Verbs'][0] = 'ComeFrom'
                
                if verbObject[ids][nodes]['Verbs'][0] == 'change':
                    verbObject[ids][nodes]['Verbs'][0] = 'changeDirection'
                    
                if verbObject[ids][k]['Verbs'][0] == 'come':
                    verbObject[ids][k]['Verbs'][0] = 'ComeFrom'
                
                if verbObject[ids][k]['Verbs'][0] == 'change':
                    verbObject[ids][k]['Verbs'][0] = 'changeDirection'
                    
#                 g.add((URIRef(clevrerHumansData+subjectUUID), RDF.type, URIRef(clevrerHumans+verbObject[ids][k]['Verbs'][0].capitalize())))
#                 g.add((URIRef(clevrerHumansData+subjectUUID), causal.causesType, URIRef(clevrerHumans+verbObject[ids][nodes]['Verbs'][0].capitalize())))
                fileTriples.writelines([URIRef(clevrerHumansData+subjectUUID),",", RDF.type,",", URIRef(clevrerHumans+verbObject[ids][k]['Verbs'][0].capitalize()),".","\n"])
                fileTriples.writelines([URIRef(clevrerHumansData+subjectUUID),",", causal.causesType,",", URIRef(clevrerHumans+verbObject[ids][nodes]['Verbs'][0].capitalize()),".","\n"])
                fileHyper.writelines([URIRef(clevrerHumansData+subjectUUID),",", RDF.type,",", URIRef(clevrerHumans+verbObject[ids][k]['Verbs'][0].capitalize()),".","\n"])
                fileHyper.writelines([URIRef(clevrerHumansData+subjectUUID),",", causal.causesType,",", URIRef(clevrerHumans+verbObject[ids][nodes]['Verbs'][0].capitalize()),".","\n"])
                
#                 g.add((URIRef(clevrerHumansData+objectUUID), RDF.type, URIRef(clevrerHumans+verbObject[ids][nodes]['Verbs'][0].capitalize())))
#                 g.add((URIRef(clevrerHumansData+objectUUID), causal.causedByType, URIRef(clevrerHumans+verbObject[ids][k]['Verbs'][0].capitalize())))
                fileTriples.writelines([URIRef(clevrerHumansData+objectUUID),",", RDF.type,",", URIRef(clevrerHumans+verbObject[ids][nodes]['Verbs'][0].capitalize()),".","\n"])
                fileTriples.writelines([URIRef(clevrerHumansData+objectUUID),",", causal.causedByType,",", URIRef(clevrerHumans+verbObject[ids][k]['Verbs'][0].capitalize()),".","\n"])
                fileHyper.writelines([URIRef(clevrerHumansData+objectUUID),",", RDF.type,",", URIRef(clevrerHumans+verbObject[ids][nodes]['Verbs'][0].capitalize()),".","\n"])
                fileHyper.writelines([URIRef(clevrerHumansData+objectUUID),",", causal.causedByType,",", URIRef(clevrerHumans+verbObject[ids][k]['Verbs'][0].capitalize()),".","\n"])

                
#                 g.add((URIRef(clevrerHumansData+subjectUUID), causal.causes, URIRef(clevrerHumansData+objectUUID)))
                fileTriples.writelines([URIRef(clevrerHumansData+subjectUUID),",", causal.causes,",", URIRef(clevrerHumansData+objectUUID),".","\n"])
                fileHyper.writelines([URIRef(clevrerHumansData+subjectUUID),",", causal.causes,",", URIRef(clevrerHumansData+objectUUID),".","\n"])

#                 g.add((URIRef(clevrerHumansData+objectUUID), causal.causedBy, URIRef(clevrerHumansData+subjectUUID)))
                fileTriples.writelines([URIRef(clevrerHumansData+objectUUID),",", causal.causedBy,",", URIRef(clevrerHumansData+subjectUUID),".","\n"])
                fileHyper.writelines([URIRef(clevrerHumansData+objectUUID),",", causal.causedBy,",", URIRef(clevrerHumansData+subjectUUID),".","\n"])

                
    rootList = list()
    [rootList.append(n) for n,d in ceg_full.in_degree() if d==0]
    
#     print("\n RootList: ",rootList)
    
    dfs = list()
    for root in rootList:
#         print("\n Root: ", root, "\n")
        l = (list(nx.dfs_edges(ceg_full, source=root, depth_limit=2))) #dfs_tree
#         print(l,"\n")
        for i in range(len(l)):
#             print(l[i])
            if l[i][0]==l[i-1][1]:
        
                if verbObject[ids][l[i][1]]['Verbs'][0] == 'come':
                    verbObject[ids][l[i][1]]['Verbs'][0] = 'ComeFrom'
                
                if verbObject[ids][l[i][1]]['Verbs'][0] == 'change':
                    verbObject[ids][l[i][1]]['Verbs'][0] = 'changeDirection'
                    
                if verbObject[ids][l[i-1][0]]['Verbs'][0] == 'come':
                    verbObject[ids][l[i-1][0]]['Verbs'][0] = 'ComeFrom'
                
                if verbObject[ids][l[i-1][0]]['Verbs'][0] == 'change':
                    verbObject[ids][l[i-1][0]]['Verbs'][0] = 'changeDirection'
                    
#
                #########HasMediatorType
                fileHyper.writelines([URIRef(clevrerHumansData+nodeDict[l[i-1][0]]),",", causal.causesType,",", URIRef(clevrerHumans+verbObject[ids][l[i][1]]['Verbs'][0].capitalize()),",", causal.hasMediator, ",", URIRef(clevrerHumansData+nodeDict[l[i][0]]),",", causal.hasMediatorType,",", URIRef(clevrerHumansData+verbObject[ids][l[i][0]]['Verbs'][0].capitalize()),".","\n"])
                fileHyper.writelines([URIRef(clevrerHumansData+nodeDict[l[i][1]]),",", causal.causedByType,",", URIRef(clevrerHumans+verbObject[ids][l[i-1][0]]['Verbs'][0].capitalize()),",", causal.hasMediator, ",", URIRef(clevrerHumansData+nodeDict[l[i][0]]),",", causal.hasMediatorType,",", URIRef(clevrerHumansData+verbObject[ids][l[i][0]]['Verbs'][0].capitalize()),".","\n"])

                fileHyper.writelines([URIRef(clevrerHumansData+nodeDict[l[i-1][0]]),",", causal.causes,",", URIRef(clevrerHumansData+nodeDict[l[i][1]]),",", causal.hasMediator, ",", URIRef(clevrerHumansData+nodeDict[l[i][0]]), ",", causal.hasMediatorType, ",", URIRef(clevrerHumansData+verbObject[ids][l[i][0]]['Verbs'][0].capitalize()),".","\n"])
                fileHyper.writelines([URIRef(clevrerHumansData+nodeDict[l[i][1]]),",", causal.causedBy,",", URIRef(clevrerHumansData+nodeDict[l[i-1][0]]),",", causal.hasMediator, ",", URIRef(clevrerHumansData+nodeDict[l[i][0]]), ",", causal.hasMediatorType, ",", URIRef(clevrerHumansData+verbObject[ids][l[i][0]]['Verbs'][0].capitalize()),".","\n"])

fileTriples.close()
fileHyper.close()

hyper=pd.read_csv("./Data/CausalCLEVRERHumans_HyperGraph.txt",names=['h','r','t','hasM1','m1'],header=None,sep=",")
triple=pd.read_csv("./Data/CausalCLEVRERHumans_Triples.txt",names=['h','r','t','hasM1','m1'],header=None,sep=",")

def traintestvalidsplit(hyper, triple,folderName):
    data = hyper.to_numpy()

    # data
    X_train_valid, X_test = train_test_split_no_unseen(data, test_size=0.1)
    X_train, X_valid = train_test_split_no_unseen(X_train_valid, test_size=0.1)

    X_train_df = pd.DataFrame(X_train, index=None, columns=None)
    X_test_df = pd.DataFrame(X_test, index=None, columns=None)
    X_valid_df = pd.DataFrame(X_valid, index=None, columns=None)
    
    X_train_df.to_csv('/StarE_code/StarE/data/clean/'+folderName+'/statements/train.txt',columns=None, header=False, index=False)
    X_test_df.to_csv('/StarE_code/StarE/data/clean/'+folderName+'/statements/test.txt',columns=None, header=False, index=False)
    X_valid_df.to_csv('/StarE_code/StarE/data/clean/'+folderName+'/statements/valid.txt',columns=None, header=False, index=False)


    data = triple.to_numpy()

    # data
    X_train_valid, X_test = train_test_split_no_unseen(data, test_size=0.1)
    X_train, X_valid = train_test_split_no_unseen(X_train_valid, test_size=0.1)

    X_train_df = pd.DataFrame(X_train, index=None, columns=None)
    X_test_df = pd.DataFrame(X_test, index=None, columns=None)
    X_valid_df = pd.DataFrame(X_valid, index=None, columns=None)
    
    X_train_df.to_csv('/StarE_code/StarE/data/clean/'+folderName+'/triples/train.txt',columns=None, header=False, index=False)
    X_test_df.to_csv('/StarE_code/StarE/data/clean/'+folderName+'/triples/test.txt',columns=None, header=False, index=False)
    X_valid_df.to_csv('/StarE_code/StarE/data/clean/'+folderName+'/triples/valid.txt',columns=None, header=False, index=False)

    return

## Causal Prediction: causesType
hyper=pd.read_csv("./Data/CausalCLEVRERHumans_HyperGraph.txt",names=['h','r','t','hasM1','m1'],header=None,sep=",")
triple=pd.read_csv("./Data/CausalCLEVRERHumans_Triples.txt",names=['h','r','t','hasM1','m1'],header=None,sep=",")
        
CC_Hyper=hyper[hyper['r'].isin(["http://semantic.bosch.com/causal/v00/causes","http://semantic.bosch.com/causal/v00/causedBy","http://semantic.bosch.com/causal/v00/causesType"])]
CC_Triple=triple[triple['r'].isin(["http://semantic.bosch.com/causal/v00/causes","http://semantic.bosch.com/causal/v00/causedBy","http://semantic.bosch.com/causal/v00/causesType"])]

CC_Hyper.to_csv("./Data/CausalCLEVRERHumans_HyperGraph_CausesCausedByCausesType.txt",columns=None, header=False, index=False)
CC_Triple.to_csv("./Data/CausalCLEVRERHumans_Triples_CausesCausedByCausesType.txt",columns=None, header=False, index=False)

traintestvalidsplit(CC_Hyper, CC_Triple, 'CLEVRERHumansPredictionHasMediatorType_C')

## Causal Explanation: causedByType
hyper=pd.read_csv("./Data/CausalCLEVRERHumans_HyperGraph.txt",names=['h','r','t','hasM1','m1'],header=None,sep=",")
triple=pd.read_csv("./Data/CausalCLEVRERHumans_Triples.txt",names=['h','r','t','hasM1','m1'],header=None,sep=",")

CC_Hyper=hyper[hyper['r'].isin(["http://semantic.bosch.com/causal/v00/causes","http://semantic.bosch.com/causal/v00/causedBy","http://semantic.bosch.com/causal/v00/causedByType"])]
CC_Triple=triple[triple['r'].isin(["http://semantic.bosch.com/causal/v00/causes","http://semantic.bosch.com/causal/v00/causedBy","http://semantic.bosch.com/causal/v00/causedByType"])]

CC_Hyper.to_csv("./Data/CausalCLEVRERHumans_HyperGraph_CausesCausedByCausedByType.txt",columns=None, header=False, index=False)
CC_Triple.to_csv("./Data/CausalCLEVRERHumans_Triples_CausesCausedByCausedByType.txt",columns=None, header=False, index=False)

traintestvalidsplit(CC_Hyper, CC_Triple, 'CLEVRERHumansExplanationHasMediatorType_C')


