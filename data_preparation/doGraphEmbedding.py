import torch
import numpy as np
import re
import h5py
import json
import pickle
import sys
sys.path.append("PyTorch-BigGraph/")
from torchbiggraph.model import  DotComparator


path_to_model = '/export2/scratch/8steinbi/data2/graph_embedding'
# nlp = spacy.load("en_core_web_md")
ENTITY_PATTERN = re.compile('Q[0-9]+')
RELATION_PATTERN = re.compile('P[0-9]+')


with open("/export2/scratch/8steinbi/data2/train_data/all_questions_trainset.json", "r") as questionFile:
    train_questions = json.load(questionFile)
print('opend all questions')
#load all paths for each startpoint per question
with open("/export2/scratch/8steinbi/data2/train_data/contextPaths_trainset_model_3.json", "r") as afile:
    train_paths = json.load(afile)

print('opend all contextPaths_trainset')

with open("/export2/scratch/8steinbi/data2/test_data/all_questions_testset.json", "r") as questionFile:
    test_questions = json.load(questionFile)

#load all paths for each startpoint per question
with open("/export2/scratch/8steinbi/data2/test_data/contextPaths_testset_model_3.json", "r") as afile:
    test_paths = json.load(afile)


def getNodeIndex(entity_kg_id, entity_names):
    if re.match(ENTITY_PATTERN,entity_kg_id):
        try:
            node_offset= entity_names.index(f"<http://www.wikidata.org/entity/{entity_kg_id}>")
            return node_offset
        except:

            print('Error Node is 0')
            return 0
    else:
        return 0
def getRelationIndex(relation_kg_id, rel_type_names):
    if re.match(ENTITY_PATTERN,relation_kg_id):
        rel_type_offset= rel_type_names.index(f"<http://www.wikidata.org/prop/direct/{relation_kg_id}>")
        return rel_type_offset
    else:
        return 0

def getHeadEmbeddings(trained_embedding,offset ):

    head_embedding = torch.from_numpy(trained_embedding["embeddings"][offset, :])
    return head_embedding

def getTailEmbeddings(trained_embedding,offset ):
    tail_embedding = torch.from_numpy(trained_embedding["embeddings"][offset, :])
    return tail_embedding

def getContextPathHeadTail(paths):
    action_labels = dict()
    for key in paths.keys():
        action_labels[key] = []
        actions = paths[key]
        for a in actions:
            p_labels = []

            p_labels.append(a[0])
            #use if relationship to be included
            #for aId in a[1]:
            #    p_labels += ut.getLabel(aId) + " "

            p_labels.append(a[2])
            action_labels[key].append(p_labels)
    return action_labels

def getHeadTailEmbedding(head_tail_dic, entity_names, trained_embedding):

    counter =0
    embeddings = dict()
    for point in head_tail_dic.keys():

        embeddings[point]=[]
        outgoingpath = head_tail_dic[point]
        print('At', counter, 'from ',len(head_tail_dic),', the point ',point,' has ', len(outgoingpath), 'pairs')
        counter += 1
        for pair in outgoingpath:
            index_head = getNodeIndex(pair[0], entity_names)
            index_tail = getNodeIndex(pair[1], entity_names)

            embedding_head = getHeadEmbeddings(trained_embedding, index_head)
            embedding_tail = getTailEmbeddings(trained_embedding, index_tail)

            sim_score = compareDotProdHeadTail(embedding_head,embedding_tail, comparator)
            int_sim_score = sim_score.item()


            embeddings[point].append(int_sim_score)

            #print(counter)





    return embeddings


def compareDotProdHeadTail(start, dest, comparator):
    '''
    create per strating Point a vector (1,1000)
    '''

    scores, _, _ = comparator(
        comparator.prepare(start.view(1, 1, 200)),
        comparator.prepare(dest.view(1, 1, 200)),

        torch.empty(1, 0, 200),  # Left-hand side negatives, not needed
        torch.empty(1, 0, 200),  # Right-hand side negatives, not needed
    )

    return scores

'''
Steps: 
1. get head indexed and encoded
2. det tail indexed and encoded
3. compare both in TransE way (dot prod) and save as [1,1000] Vector for each starting Point

'''
comparator = DotComparator()

#get the head and tail for each startingPoint
#train_headtail = getContextPathHeadTail(train_paths)

#test_headtail = getContextPathHeadTail(test_paths)



with open("/export2/scratch/8steinbi/data2/train_data/headtail_labels_trainset.json", "r") as qfile:

    train_headtail =json.load( qfile)


print('Headtail loaded')

with open("/export2/scratch/8steinbi/data2/test_data/headtail_labels_testset.json", "r") as qfile:
   test_headtail = json.load(qfile)

#get Embedding foor tail and head:


print('Length of Train Embedding ',len(train_headtail))
print('Length of Test Embedding ',len(test_headtail))

with open(f"{path_to_model}/entity_names_all_0.json", "rt") as tf:
    entity_names = json.load(tf)

with h5py.File(f"{path_to_model}/embeddings_all_0.v0.h5", "r") as hf:
    train_embedding = getHeadTailEmbedding(train_headtail,entity_names, hf)
    test_embedding = getHeadTailEmbedding(test_headtail,entity_names,hf)


with open("/export2/scratch/8steinbi/data2/train_data/embedded_sim_score_vector_model3.json", "w") as q_file:
    json.dump(train_embedding, q_file)

with open("/export2/scratch/8steinbi/data2/train_data/embedded_sim_score_vector_model3.pickle", "wb") as q_file:
    pickle.dump(train_embedding, q_file)
print('saved Train  embedding')


with open("/export2/scratch/8steinbi/data2/test_data/embedded_sim_score_vector_model3.json", "w") as q_file:
    json.dump(test_embedding, q_file)

with open("/export2/scratch/8steinbi/data2/test_data/embedded_sim_score_vector_model3.pickle", "wb") as q_file:
    pickle.dump(test_embedding, q_file)
#print('saved Test  embedding')







