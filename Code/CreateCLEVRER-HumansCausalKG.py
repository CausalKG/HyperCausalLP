# ##### Create a KG
import rdflib
from rdflib import URIRef
from rdflib.namespace import Namespace, NamespaceManager
from rdflib.namespace import RDF, RDFS
from rdflib import Graph
import uuid

with open('./Data/CegWithNoCycles.txt', 'r') as f:
    cegWithNoCycles = json.load(f)

with open('./Data/VerbObject.json', 'r') as f:
    verbObject = json.load(f)

file = open('./Data/valid_ceg_data_May12.p', 'rb')  
validData = pickle.load(file)
file.close()

file = open('./Data/train_ceg_data_May12.p', 'rb')  
trainData = pickle.load(file)
file.close()

data = ChainMap(trainData, validData)
 
file=open("./Data/CausalClevrerHumanKG.txt","w")

clevrerHumans = Namespace("http://semantic.bosch.com/CausalClevrerHumans/v00/")
clevrerHumansData = Namespace("http://semantic.bosch.com/CausalClevrerHumans/v00/data/")
clevrer = Namespace("http://semantic.bosch.com/CLEVRER/")
causal = Namespace("http://semantic.bosch.com/causal/v00/")
scene = Namespace("http://semantic.bosch.com/scene/v02/")
ssn = Namespace("http://www.w3.org/ns/ssn/")

g = Graph()
namespace_manager = NamespaceManager(Graph())
namespace_manager.bind('', clevrerHumansData, override=False)
namespace_manager.bind("causal", causal, override=True)
namespace_manager.bind("CCH", clevrerHumans, override=True)
namespace_manager.bind("scene", scene, override=True)
namespace_manager.bind("ssn", ssn, override=True)
g.namespace_manager = namespace_manager

# Add for the Shape, color and material property to the graph
# Shape: Cube, Sphere, Ball, 
# Get unique shape, color, and material
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
    g.add((colorUUID, RDF.type, URIRef(clevrerHumans+"Color"))) 
    g.add((colorUUID, RDFS.label, rdflib.term.Literal(c))) 
    file.writelines([colorUUID,",", RDF.type,",", URIRef(clevrerHumans+"Color"),".","\n"])
    file.writelines([colorUUID,",", RDFS.label,",", rdflib.term.Literal(c),".","\n"])

for s in shapeSet:
    shapeUUID = URIRef(clevrerHumansData+str(uuid.uuid4()))  
    shape[s] = shapeUUID
    g.add((shapeUUID, RDF.type, URIRef(clevrerHumans+"Shape"))) 
    g.add((shapeUUID, RDFS.label, rdflib.term.Literal(s))) 
    file.writelines([shapeUUID,",", RDF.type,",", URIRef(clevrerHumans+"Shape"),".","\n"])
    file.writelines([shapeUUID,",", RDFS.label,",", rdflib.term.Literal(s),".","\n"])

for m in materialSet:
    materialUUID = URIRef(clevrerHumansData+str(uuid.uuid4()))   
    material[m] = materialUUID
    g.add((materialUUID, RDF.type, URIRef(clevrerHumans+"Material")))  
    g.add((materialUUID, RDFS.label, rdflib.term.Literal(m))) 
    file.writelines([materialUUID,",", RDF.type,",", URIRef(clevrerHumans+"Material"),".","\n"])
    file.writelines([materialUUID,",", RDFS.label,",", rdflib.term.Literal(m),".","\n"])

objectParticipant = dict()
for c in color:
    for s in shape:
        objectParticipantUUID = URIRef(clevrerHumansData+str(uuid.uuid4()))             
        objectParticipant[c+"_"+s] = objectParticipantUUID
        g.add(((objectParticipantUUID), RDF.type, scene.Object)) 
        g.add(((objectParticipantUUID), RDFS.label, rdflib.term.Literal(c+" "+s))) 
        file.writelines([objectParticipantUUID,",", RDF.type,",", scene.Object,".","\n"])
        file.writelines([objectParticipantUUID,",", RDFS.label,",", rdflib.term.Literal(c+" "+s),".","\n"])
        
        g.add(((objectParticipantUUID), ssn.hasProperty, color[c]))
        g.add(((objectParticipantUUID), ssn.hasProperty, shape[s]))

        file.writelines([objectParticipantUUID,",", ssn.hasProperty,",", color[c],".","\n"])
        file.writelines([objectParticipantUUID,",", ssn.hasProperty,",", shape[s],".","\n"])

        
for c in color:
    for s in shape:
        for m in material:
            if c not in ['silver', 'gold']:
                objectParticipantUUID = URIRef(clevrerHumansData+str(uuid.uuid4()))             
                objectParticipant[c+"_"+m+"_"+s] = objectParticipantUUID
                g.add(((objectParticipantUUID), RDF.type, scene.Object)) 
                g.add(((objectParticipantUUID), RDFS.label, rdflib.term.Literal(c+" "+m+" "+s))) 
                file.writelines([objectParticipantUUID,",", RDF.type,",", scene.Object,".","\n"])
                file.writelines([objectParticipantUUID,",", RDFS.label,",", rdflib.term.Literal(c+" "+m+" "+s),".","\n"])
        
                g.add(((objectParticipantUUID), ssn.hasProperty, color[c]))
                g.add(((objectParticipantUUID), ssn.hasProperty, shape[s]))
                g.add(((objectParticipantUUID), ssn.hasProperty, material[m]))
                
                file.writelines([objectParticipantUUID,",", ssn.hasProperty,",", color[c],".","\n"])
                file.writelines([objectParticipantUUID,",", ssn.hasProperty,",", shape[s],".","\n"])
                file.writelines([objectParticipantUUID,",", ssn.hasProperty,",", material[m],".","\n"])

                
for ids in cegWithNoCycles:
    g.add((URIRef(clevrerHumansData+(ids)), RDF.type, scene.Scene))
    file.writelines([URIRef(clevrerHumansData+(ids)),",", RDF.type,",", scene.Scene,".","\n"])

    
    ceg_data = data[ids]
    ceg_full=ceg_data.get('CEG_full')
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
                
                g.add((URIRef(clevrerHumansData+(ids)), scene.includes, URIRef(clevrerHumansData+subjectUUID))) 
                g.add((URIRef(clevrerHumansData+subjectUUID), RDFS.label, rdflib.term.Literal(cegSubjectNode))) 
                file.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", URIRef(clevrerHumansData+subjectUUID),".","\n"])
                file.writelines([URIRef(clevrerHumansData+subjectUUID),",", RDFS.label,",", rdflib.term.Literal(cegSubjectNode),".","\n"])
               
                g.add((URIRef(clevrerHumansData+(ids)), scene.includes, URIRef(clevrerHumansData+objectUUID))) 
                g.add((URIRef(clevrerHumansData+objectUUID), RDFS.label, rdflib.term.Literal(cegObjectNode))) 
                file.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", URIRef(clevrerHumansData+objectUUID),".","\n"])
                file.writelines([URIRef(clevrerHumansData+objectUUID),",", RDFS.label,",", rdflib.term.Literal(cegObjectNode),".","\n"])
               
                for objects in verbObject[ids][k]['Object']:
                    if objects:
                        if 'Material' in objects.keys():
#                             print(ids,k,objects, objects.keys(),objects['Material'])
#                             print(objects['Color'], objects['Material'], objects['Shape'])
                            uid = objectParticipant[objects['Color']+"_"+objects['Material']+"_"+objects['Shape']]
                            
                            g.add((URIRef(clevrerHumansData+(ids)), scene.includes, uid))
                            file.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", uid,".","\n"])
                            
                            g.add((URIRef(clevrerHumansData+subjectUUID), scene.hasParticipant, uid))
                            g.add((uid, scene.isParticipantIn, URIRef(clevrerHumansData+subjectUUID)))
                            file.writelines([URIRef(clevrerHumansData+subjectUUID),",", scene.hasParticipant,",", uid,".","\n"])
                            file.writelines([uid,",", scene.isParticipantIn,",", URIRef(clevrerHumansData+subjectUUID),".","\n"])
                              
                        else:
                            uid = objectParticipant[objects['Color']+"_"+objects['Shape']]
                            
                            g.add((URIRef(clevrerHumansData+(ids)), scene.includes, uid))
                            file.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", uid,".","\n"])
            
                            g.add((URIRef(clevrerHumansData+subjectUUID), scene.hasParticipant, uid))
                            g.add((uid, scene.isParticipantIn, URIRef(clevrerHumansData+subjectUUID)))
                            file.writelines([URIRef(clevrerHumansData+subjectUUID),",", scene.hasParticipant,",", uid,".","\n"])
                            file.writelines([uid,",", scene.isParticipantIn,",", URIRef(clevrerHumansData+subjectUUID),".","\n"])
                    
                                
                    
                for objects in verbObject[ids][nodes]['Object']:
                    if objects:
                        if 'Material' in objects.keys():
                            uid = objectParticipant[objects['Color']+"_"+objects['Material']+"_"+objects['Shape']]
                            
                            g.add((URIRef(clevrerHumansData+(ids)), scene.includes, uid))
                            file.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", uid,".","\n"])
                             
                            g.add((URIRef(clevrerHumansData+objectUUID), scene.hasParticipant, uid))
                            g.add((uid, scene.isParticipantIn, URIRef(clevrerHumansData+objectUUID)))
                            file.writelines([URIRef(clevrerHumansData+objectUUID),",", scene.hasParticipant,",", uid,".","\n"])
                            file.writelines([uid,",", scene.isParticipantIn,",", URIRef(clevrerHumansData+objectUUID),".","\n"])
                
                        else:
                            uid = objectParticipant[objects['Color']+"_"+objects['Shape']]
                            
                            g.add((URIRef(clevrerHumansData+(ids)), scene.includes, uid))
                            file.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", uid,".","\n"])
                             
                            g.add((URIRef(clevrerHumansData+objectUUID), scene.hasParticipant, uid))
                            g.add((uid, scene.isParticipantIn, URIRef(clevrerHumansData+objectUUID)))
                            file.writelines([URIRef(clevrerHumansData+objectUUID),",", scene.hasParticipant,",", uid,".","\n"])
                            file.writelines([uid,",", scene.isParticipantIn,",", URIRef(clevrerHumansData+objectUUID),".","\n"])

                
                if verbObject[ids][nodes]['Verbs'][0] == 'come':
                    verbObject[ids][nodes]['Verbs'][0] = 'ComeFrom'
                
                if verbObject[ids][nodes]['Verbs'][0] == 'change':
                    verbObject[ids][nodes]['Verbs'][0] = 'changeDirection'
                    
                if verbObject[ids][k]['Verbs'][0] == 'come':
                    verbObject[ids][k]['Verbs'][0] = 'ComeFrom'
                
                if verbObject[ids][k]['Verbs'][0] == 'change':
                    verbObject[ids][k]['Verbs'][0] = 'changeDirection'
                
                g.add((URIRef(clevrerHumansData+subjectUUID), RDF.type, URIRef(clevrerHumans+verbObject[ids][k]['Verbs'][0].capitalize())))
                g.add((URIRef(clevrerHumansData+subjectUUID), causal.causesType, URIRef(clevrerHumans+verbObject[ids][nodes]['Verbs'][0].capitalize())))
                file.writelines([URIRef(clevrerHumansData+subjectUUID),",", RDF.type,",", URIRef(clevrerHumans+verbObject[ids][k]['Verbs'][0].capitalize()),".","\n"])
                file.writelines([URIRef(clevrerHumansData+subjectUUID),",", causal.causesType,",", URIRef(clevrerHumans+verbObject[ids][nodes]['Verbs'][0].capitalize()),",",str(ceg_full.get_edge_data(k, nodes)['weight']),".","\n"])
                
                g.add((URIRef(clevrerHumansData+objectUUID), RDF.type, URIRef(clevrerHumans+verbObject[ids][nodes]['Verbs'][0].capitalize())))
                g.add((URIRef(clevrerHumansData+objectUUID), causal.causedByType, URIRef(clevrerHumans+verbObject[ids][k]['Verbs'][0].capitalize())))
                file.writelines([URIRef(clevrerHumansData+objectUUID),",", RDF.type,",", URIRef(clevrerHumans+verbObject[ids][nodes]['Verbs'][0].capitalize()),".","\n"])
                file.writelines([URIRef(clevrerHumansData+objectUUID),",", causal.causedByType,",", URIRef(clevrerHumans+verbObject[ids][k]['Verbs'][0].capitalize()),",",str(ceg_full.get_edge_data(k, nodes)['weight']),".","\n"])

                
                g.add((URIRef(clevrerHumansData+subjectUUID), causal.causes, URIRef(clevrerHumansData+objectUUID)))
                file.writelines([URIRef(clevrerHumansData+subjectUUID),",", causal.causes,",", URIRef(clevrerHumansData+objectUUID),",",str(ceg_full.get_edge_data(k, nodes)['weight']),".","\n"])

                g.add((URIRef(clevrerHumansData+objectUUID), causal.causedBy, URIRef(clevrerHumansData+subjectUUID)))
                file.writelines([URIRef(clevrerHumansData+objectUUID),",", causal.causedBy,",", URIRef(clevrerHumansData+subjectUUID),",",str(ceg_full.get_edge_data(k, nodes)['weight']),".","\n"])

                
g.serialize('./Data/CausalClevrerHumanKG.ttl', format="n3")                             
print(len(g))
file.close()

# Average the causal weight for same event type
tmp3videos = pd.read_csv('./Data/CausalCLEVRERHumanKG.txt',names=[0,1,2,3], header=None, sep=",")

tmp3videosCauseTypeCausedByType = tmp3videos.loc[tmp3videos[1].isin(['http://semantic.bosch.com/causal/v00/causes','http://semantic.bosch.com/causal/v00/causedBy','http://semantic.bosch.com/causal/v00/causesType', 'http://semantic.bosch.com/causal/v00/causedByType'])]

duplicateTCEMean = duplicateTCE.groupby([0,1,2])[3].mean()

tmp3videos.drop(duplicateTCE.index, axis=0, inplace=True, errors = 'ignore')

for i, v in duplicateTCEMean.items():
    tmp3videos = tmp3videos.append({0:i[0],1:i[1],2:i[2],3:v}, ignore_index=True)

tmp3videos.to_csv('./Data/CausalCLEVRERHumanKG_AvgWeight.txt', index=False)
