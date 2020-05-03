import pickle
import re
import time

import numpy as np

# below section was used previously to parse and prepare the dictionaries and variables required but was later
# replaced with a pickle load method for ease
# ===================================
def corpus_reader(param):
    pattern = '((?<=\(\')[^\']*(?=\',)|(?<=\(\")[^\"]*(?=\",))'

    sentences = []

    with open( param ) as f:
        corpus = f.readlines()

    for line in corpus:
        temp = re.findall( pattern, line )
        sentences.append( temp )

    return sentences


def initial_prob_reader(param):
    tags = 'Begin - (.+?) :'
    probs = 'Begin .+ ([-+]?\d*\.\d+|\d+)'
    with open( param ) as f:
        corpus = f.read()

    temp1 = re.findall( tags, corpus )
    temp2 = re.findall( probs, corpus )

    assert len( temp1 ) == len( temp2 )
    init_prob = dict()
    total = 0
    for itr in range( len( temp2 ) ):
        total += float( temp2[itr] )
        init_prob[temp1[itr]] = float( temp2[itr] )

    #  Below normalises the initial probabilites

    max = 0
    sum = 0

    for itr in range( len( temp2 ) ):
        init_prob[temp1[itr]] = float( temp2[itr] ) / total

        temp = float( temp2[itr] ) / total
        sum += temp
        max = max if max > temp else temp

    return init_prob


def state_reader(param):
    with open( param ) as f:
        corpus = f.read()

    temp = corpus.split( "\t" )
    return temp


def transitions_reader(param):
    with open( param ) as f:
        corpus = f.readlines()
    trans_p = dict()
    total = 0

    for line in corpus:
        temp = line.split( ' ' )
        ini = temp[0]
        end = temp[2]
        temp = temp[4].split( '\n' )
        num = float( temp[0] )
        total += num
        if ini == "Begin":
            continue
        if ini not in trans_p.keys():
            trans_p[ini] = {end: num}
        else:
            trans_p[ini].update( {end: num} )

    for k1, v1 in trans_p.items():
        total = 0.0

        for k2, v2 in v1.items():
            total += v2

        for k2, v2 in v1.items():
            trans_p[k1][k2] = (v2/total)

    return trans_p


def emission_reader(param):
    tags = '\((.+?)\) ='
    probs = ' = ([-+]?\d*\.\d+|\d+)'
    with open( param ) as f:
        corpus = f.read()

    temp1 = re.findall( tags, corpus )
    temp2 = re.findall( probs, corpus )
    assert len( temp1 ) == len( temp2 )

    emi_p = dict()

    for itr in range( len( temp1 ) ):
        pair = temp1[itr].split( '|' )
        end = pair[1]
        start = pair[0]
        num = float( temp2[itr] )
        if start not in emi_p.keys():
            emi_p[start] = {end: num}
        else:
            emi_p[start].update( {end: num} )

    min = 10 # probabilities will be lower so its ok

    for k1, v1 in emi_p.items():
        total = 0.0

        for k2, v2 in v1.items():
            total += v2

        for k2, v2 in v1.items():
            emi_p[k1][k2] = (v2/total)

            if emi_p[k1][k2] < min:
                min = emi_p[k1][k2]

    global min_emission_prob
    min_emission_prob = min
    return emi_p


# ======================================

min_emission_prob = 0.0

def viterbi(obs, states, start_p, trans_p, emit_p):
    d = np.zeros( [len( states ), len( obs )] )

    # d[i][j] is max probability of sequence that end with state i and emits the observation j

    # Step 1: Set up the initialization
    for i, state in enumerate( states ):
        try:
            d[i][0] = start_p[state] * emit_p[obs[0]][state]
        except KeyError:  # for tags that cannot be the beginning of a sentence
            d[i][0] = 0

    # Step 2: Recurse and fill up matrix
    for t in range( 1, len( obs ) ):
        for j, curr_state in enumerate( states ):
            for i, prev_state in enumerate( states ):
                try:
                    val = d[i][t - 1] * trans_p[prev_state][curr_state] * emit_p[obs[t]][curr_state]
                except KeyError:  # the transition/emission doesnt exist
                    if prev_state in transitions.keys() and curr_state in transitions[prev_state].keys(): #if it is an unseen word, we give it
                        val = d[i][t - 1] * trans_p[prev_state][curr_state] * min_emission_prob
                    else:
                        continue
                if val >= d[j][t]:
                    d[j][t] = val

    # Step 3: Traceback for the best state sequence
    qs = []
    for t in reversed( range( len( obs ) ) ):
        best_state = None
        best_state_val = 0
        for i, prev_state in enumerate( states ):
            transition_prob = 1
            if t < len( obs ) - 1:
                try:
                    transition_prob = trans_p[prev_state][qs[0]]
                except KeyError:
                    continue
            if d[i][t] * transition_prob > best_state_val:
                best_state_val = d[i][t] * transition_prob
                best_state = prev_state
        qs.insert( 0, best_state )

    return qs


def get_infile():
    sentences=[]
    loop=1
    while(loop):
        try:
            sent=input().split('\n')
            sent=sent[0].split(' ')
            sentences.append([i for i in sent if i !=''])

        except EOFError:
            loop=0
    return sentences

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    observed_sentences=get_infile()
    # observed_sentences=corpus_reader("./corpus.txt")
    initial_prob_dict = load_obj("init_prob")
    states = load_obj( "states" )
    transitions = load_obj( "trans_prob" )
    emission_prob = emission_reader( './hmmmodel.txt' )


    output_sequences = []
    # Todo i changed \\/ to \/ in the corpus

    for sent in observed_sentences:
        output = viterbi( sent, states, initial_prob_dict, transitions, emission_prob )
        output_sequences.append( output )

    # with open( 'outfile.txt', 'w' ) as file:
    #     file.writelines( '  '.join( str( j ) for j in i ) + '\n' for i in output_sequences )

    for sent in output_sequences:
        for word in sent:
            print(word," ",end='')
        print()

