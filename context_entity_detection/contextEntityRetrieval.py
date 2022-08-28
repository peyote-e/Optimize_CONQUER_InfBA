import json
import re
import time

import torch
import spacy
from neo4jDatabaseConnection import KGENVIRONMENT
import numpy as np
import argparse
import sys
sys.path.append("Optimize_CONQUER_InfBA/")
sys.path.append("/export/home/8steinbi/Bachelor/Optimize_CONQUER_InfBA/BLINK/")
sys.path.append("/export/home/8steinbi/Bachelor/Optimize_CONQUER_InfBA/BLINK/elq")
sys.path.append("Optimize_CONQUER_InfBA/context_entity_detection")
sys.path.append("../utils")

start_time = time.time()
import utils as ut

sys.path.append("../BLINK/")

import BLINK.elq.main_dense as main_dense
from sentence_transformers import SentenceTransformer, util


"""Get context entities per question along KG paths starting from there;
Candidate entities are scored by four different scores (lexical match, neighbor overlap, ned score, kg prior);
ELQ is used as NED tool"""

ENTITY_PATTERN = re.compile('Q[0-9]+')
nlp = spacy.load("en_core_web_md")

# datastructures to store retrieved context entities and its KG paths
startPoints = dict()
globalSeenContextNodes = dict()



# neo4j database access
env = KGENVIRONMENT()

# best hyperparameters found:
start_score = 0.47
sim_score = 0.4
overlap_score = 0.1
prior_score = 0.15
ned_score = 0.35
bonus_score = 0.0

value_list = []
# cut off for KG prior
MAX_PRIOR = 100

# models for semantic simalarity
#model_1 = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1')
model_name = 'model_2'
model_2 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# model_3 = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1')
# model_4 = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')


with open("../data/labels_dict.json") as labelFile:
    labels_dict = json.load(labelFile)

print("labels_dict.json  loaded")

with open("../data/ConvRef/ConvRef_trainset.json") as json_file:
    train_data = json.load(json_file)
print("ConvRef_trainset.json  loaded")
with open("../data/ConvRef/ConvRef_devset.json") as json_file:
    dev_data = json.load(json_file)
print("/ConvRef_devset.jso  loaded")
with open("../data/ConvRef/ConvRef_testset.json") as json_file:
    test_data = json.load(json_file)
print(" ConvRef_testset.json  loaded")

# data+config relevant for ELQ NED
models_path = "../BLINK/models/"  # the path where you stored the ELQ models
config = {
    "interactive": False,
    "biencoder_model": models_path + "elq_wiki_large.bin",
    "biencoder_config": models_path + "elq_large_params.txt",
    "cand_token_ids_path": models_path + "entity_token_ids_128.t7",
    "entity_catalogue": models_path + "entity.jsonl",
    "entity_encoding": models_path + "all_entities_large.t7",
    "output_path": "logs/",  # logging directory
    "faiss_index": "hnsw",
    "index_path": models_path + "faiss_hnsw_index.pkl",
    "num_cand_mentions": 10,
    "num_cand_entities": 10,
    "threshold_type": "joint",
    "threshold": -4.5,
}
args = argparse.Namespace(**config)
models = main_dense.load_models(args, logger=None)
id2wikidata = json.load(open("models/id2wikidata.json"))


# datastructure for storing information about context entities for current conversation
class ContextNode:

    def __init__(self, turn):
        self.turn = turn

    def getTurn(self):
        return self.turn

    def updateTurn(self, turn):
        self.turn = turn

    def getNeighbors(self):
        return self.neighbors

    def setNeighbors(self, neighbors):
        self.neighbors = neighbors

    def getOneHopPaths(self):
        return self.oneHopPaths

    def setOneHopPaths(self, paths):
        self.oneHopPaths = paths

    def __str__(self):
        return "turn: " + str(self.turn) + ", neighbors: " + str(self.neighbors) + ", oneHopPaths: " + str(
            self.oneHopPaths)

    # calculate Bert similarity for node label and question


def getStringSimQuestionNode(nodeLabelList, question, similarity_model):
    start_time =time.time()
    try:
        sent = nlp(question)
        question_token = [token.text.lower() for token in sent if not token.is_stop]
        nodes = []
        for neighbour in nodeLabelList:
            node = ut.getLabel(neighbour)
            nodeSent = nlp(node)
            node_tokens = [token.text.lower() for token in nodeSent if not token.is_stop]
            nodes.append(' '.join(node_tokens))
        node_embeddings = similarity_model.encode(nodes)
        sentence_embeddings = similarity_model.encode([' '.join(question_token)])
        cosine_scores = util.cos_sim(sentence_embeddings[0], node_embeddings)[0].cpu().tolist()
        stop_time = time.time()
        #print('Bert Sim Measurement for '+ str(len(nodeLabelList)) + ' neigbours took ', stop_time-start_time)
        return dict(zip(nodeLabelList,cosine_scores))
    except:
        zero_list = [0]*len(nodeLabelList)
        return dict(zip(nodeLabelList,zero_list))


def get_question_type(question):

    date_type_question_beginnings = ['in what year', 'in which year', 'when was the', 'what year was', 'what year did','on what date']

    gpe_type_question_beginnings = ['in which country', 'where is the', 'where was the', 'where is the', 'what country is', 'what city was', 'in which city']

    if ' '.join(question.split()[:3]) in date_type_question_beginnings:
        return ['DATE', 'TIME']
    if ' '.join(question.split()[:3]) in gpe_type_question_beginnings:
        return ['GPE','Location']

    else:
        return ['NO_TYPE']

# find further context entities for given question in one hop neighborhood
def expandStartingPoints(context_nodes, currentid, question, tagged_entities):
    candidates = dict()
    #question_type = get_question_type(question.lower())
    # go over existing context entity for this conversation so far:
    for node in context_nodes.keys():

        neighbors = context_nodes[node].getNeighbors()
        bert_cosine_results = getStringSimQuestionNode(neighbors, question, model_2)
        time_count=0
        time_nerd = 0
        time_overlap=0
        #make out of tagged_entities a dic so ned score can be faster returned
        ned_score_dic = {entity[0]:entity[1] for entity in tagged_entities}
        # go over 1 hop neighbors
        for neighbor in neighbors:
            if len(neighbor) < 2:
                continue
            if not re.match(ENTITY_PATTERN, neighbor):
                continue
            if neighbor in context_nodes.keys():
                continue
            # check if candidate is in neighborhood of several context entities
            start_time_overlap = time.time()
            if neighbor in candidates.keys():
                candidates[neighbor]["count"] += 1
                continue
            candidates[neighbor] = dict()
            candidates[neighbor]["count"] = 1.0

            #print('OVERLAP SCORE: ',candidates[neighbor]["count"])
            stop_time_overlap=time.time()
            time_overlap += stop_time_overlap-start_time_overlap

            # add QuestionType bonus
            #if len(nlp(neighbor).ents) > 0 and nlp(neighbor).ents[0].label_ in question_type:
            #    candidates[neighbor]["bonus"] = 1.0
            #else:
            #    candidates[neighbor]["bonus"] = 0

            # use number of triples where neighbor appears as subject (= number of outgoing paths) from KG (neo4j database) as KG prior
            start_time_count = time.time()
            neighbor_count = env.get_number_of_neighbors(neighbor)

            # this is cut off and normalized by MAX_PRIOR
            if neighbor_count > MAX_PRIOR:
                candidates[neighbor]["prior"] = 1.0
            else:
                candidates[neighbor]["prior"] = neighbor_count / MAX_PRIOR
            stop_time_count = time.time()
            time_count += stop_time_count-start_time_count
            #print('NEIGHOURHOOD COUNT: ', candidates[neighbor]["prior"])
            # calculate sim between candidate and question

            candidates[neighbor]["sim"]= bert_cosine_results.get(neighbor)
            # check if candidate was discovered by NED tool
            start_time_nerd = time.time()
            tagged = False
            if neighbor in ned_score_dic:
                tagged = True
                candidates[neighbor]["nerd"] = ned_score_dic.get(neighbor)
                break
            if not tagged:
                candidates[neighbor]["nerd"] = 0.0
            #print('NERD SCORE: ', candidates[neighbor]["nerd"])
            stop_time_nerd = time.time()
            time_nerd += stop_time_nerd-start_time_nerd

    new_starts = []
    for candidate in candidates.keys():
        # normalize count by number of total context nodes we have
        candidates[candidate]["count"] /= len(context_nodes)
        # calculate score consisting of four different scores (similarity, neighborhood overlap, KG prior and NED score)
        score = sim_score * candidates[candidate]["sim"] + overlap_score * candidates[candidate]["count"] + prior_score * candidates[candidate]["prior"] + ned_score * candidates[candidate]["nerd"]
        value_list.append({'candidate': candidate, 'question': question, 'sim':candidates[candidate]["sim"] , 'count': candidates[candidate]["count"], 'prior': candidates[candidate]["prior"], 'ned': candidates[candidate]["nerd"]})
        # only take candidates above the threshold
        if score >= start_score:
            new_starts.append(candidate)
            print('new starts added :' + str(candidate))

    #print('Overlap calc took avrg. ', time_overlap/len(neighbors))
    #print('Time for neighbour prior took ', time_count/len(neighbors))
    #print('Time to calc avrg nerd: ', time_nerd/len(neighbors))
    return new_starts


def retrieveContextEntities(question, currentid, turn, context_nodes, elq_predictions):
    # context nodes from previous turns can be also relevant for current one:
    startPoints[currentid] = list(context_nodes.keys())

    # check if we don't have any context entities yet, then use the ones predicted from the NED tool as initial entities
    if (currentid.endswith("-0") and not currentid.count("-") == 2) or not context_nodes:
        for entry in elq_predictions:
            if not entry[0] in startPoints[currentid]:
                startPoints[currentid].append(entry[0])
            updateContext(context_nodes, entry[0], turn)

    # score entities in one hop neighborhood and retrieve new context nodes
    new_starts = expandStartingPoints(context_nodes, currentid, question, elq_predictions)
    for newId in new_starts:
        updateContext(context_nodes, newId, turn)

        if not newId in startPoints[currentid]:
            startPoints[currentid].append(newId)

    return


def updateContext(context_nodes, newId, turn):
    if len(newId) < 2:
        return
    # only add entities to context
    if not re.match(ENTITY_PATTERN, newId):
        return
    # check if we already have id in context
    if newId in context_nodes.keys():
        return
    newNode = ContextNode(turn)
    # get paths starting from entity
    if newId in globalSeenContextNodes.keys():
        paths = globalSeenContextNodes[newId]
    else:
        # for new ids: get paths from neo4j database
        paths = env.get_one_hop_nodes(newId)
        globalSeenContextNodes[newId] = paths
    neighbors = []
    # neighbors are nodes in one hop distance
    for path in paths:
        neighbors.append(path[-1])

    newNode.setNeighbors(list(set(neighbors)))
    context_nodes[newId] = newNode
    return


def getElqPredictions(question_id, convquestion):
    data_to_link = [{"id": question_id, "text": convquestion}]
    # run elq to get predictions for current conversational question
    predictions = main_dense.run(args, None, *models, test_data=data_to_link)
    elq_predictions = []
    for prediction in predictions:
        pred_scores = prediction["scores"]
        # get entity ids from wikidata
        pred_ids = [id2wikidata.get(wikipedia_id) for (wikipedia_id, a, b) in prediction['pred_triples']]
        p = 0
        for predId in pred_ids:
            if predId is None:
                continue
            # normalize the score
            score = np.exp(pred_scores[p])
            i = 0
            modified = False
            # potentially update score if same entity is matched multiple times
            for tup in elq_predictions:
                if tup[0] == predId:
                    modified = True
                    if score > tup[1]:
                        elq_predictions[i] = (predId, score)
                i += 1
            # store enitity id along its normalized score
            if not modified:
                elq_predictions.append((predId, score))
            p += 1

    return elq_predictions


def processData(data):
    # go over each question/reformulation in ConvRef to retrieve the context entities
    timer= 0
    counter = 0
    for conv in data:
        context_nodes = dict()
        convquestion = ""
        # go over each question
        start_time_questionset = time.time()
        for question_info in conv['questions']:

            elq_predictions = dict()
            question_id = question_info['question_id']
            turn = int(question_id.split("-")[1])
            question = question_info["question"]
            convquestion += question_info["question"] + " "
            # get predictions from NED tool as one scoring factor, include conv. history for better results
            elq_predictions = getElqPredictions(question_id, convquestion)
            print('The elq pred. : ', elq_predictions )
            retrieveContextEntities(question, question_id, turn, context_nodes, elq_predictions)

            # go over each available reformulation
            for ref_info in question_info["reformulations"]:
                ref_id = ref_info["ref_id"]
                reformulation = ref_info["reformulation"]
                # get predictions from NED tool as one scoring factor, include conv. history for better results
                elq_predictions = getElqPredictions(question_id, convquestion + " " + reformulation)
                retrieveContextEntities(reformulation, ref_id, turn, context_nodes, elq_predictions)

        counter += 1
        stop_time_questionset=time.time()
        time_it_took= stop_time_questionset-start_time_questionset
        timer+= time_it_took
        print('Questionset took : ', time_it_took/60, ' min')
        print(counter)
        if counter == 50:
            print('Avrg. Time Questionset took: ', timer/50)
            timer = 0
            counter = 0
            print("question id: ", question_id, ", question: ", question)
            print(question_id ," of ", len(data) )
            with open(f"/export/home/8steinbi/data/train_data/entity_values_{question_id}.json", "w") as check_file:
                json.dump(startPoints, check_file)
            sys.stdout.flush()
    return


if __name__ == '__main__':
    processData(train_data)
    #store entity values

    with open(f"/export/home/8steinbi/data/train_data/entity_values_{model_name}.json", "w") as value_file:
        json.dump(value_list,value_file)
    # store startpoints per question
    with open(f"/export/home/8steinbi/data/train_data/startPoints_trainset_{model_name}.json", "w") as start_file:
        json.dump(startPoints, start_file)

    # store paths per startpoint
    with open(f"/export/home/8steinbi/data/train_data/contextPaths_trainset._{model_name}json", "w") as path_file:
        json.dump(globalSeenContextNodes, path_file)

    print("Successfully stored startpoints and KG paths for training data")

    # reset startpoints and contextpaths
    startPoints = dict()
    globalSeenContextNodes = dict()
    processData(dev_data)

    # store startpoints per question
    with open(f"/export/home/8steinbi/data/dev_data/startPoints_devset_{model_name}.json", "w") as start_file:
        json.dump(startPoints, start_file)

    # store paths per startpoint
    with open(f"/export/home/8steinbi/data/dev_data/contextPaths_devset_{model_name}.json", "w") as path_file:
        json.dump(globalSeenContextNodes, path_file)

    print("Successfully stored startpoints and KG paths for dev data")

    # reset startpoints and contextpaths
    startPoints = dict()
    globalSeenContextNodes = dict()
    processData(test_data)

    # store startpoints per question
    with open(f"/export/home/8steinbi/data/test_data/startPoints_testset_{model_name}.json", "w") as start_file:
        json.dump(startPoints, start_file)

    # store paths per startpoint
    with open(f"/export/home/8steinbi/data/test_data/contextPaths_testset_{model_name}.json", "w") as path_file:
        json.dump(globalSeenContextNodes, path_file)

    print("Successfully stored startpoints and KG paths for test data")

    end_time= time.time()
    print('Run time was'+ str(end_time-start_time) +' sec')