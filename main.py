import re
import time

import numpy as np


def viterbi(obs, states, start_p, trans_p, emit_p):
    d = np.zeros( [len( states ), len( obs )] )

    # d[i][j] is max probability of sequence that end with state i and emits the observation j

    # Step 1: Set up the initialization
    for i, state in enumerate( states ):
        try:
            d[i][0] = start_p[state] * emit_p[state][obs[0]]
        except KeyError:  # for tags that cannot be the beginning of a sentence
            d[i][0] = 0

    # Step 2: Recurse and fill up matrix
    for t in range( 1, len( obs ) ):
        for j, curr_state in enumerate( states ):
            for i, prev_state in enumerate( states ):
                try:
                    val = d[i][t - 1] * trans_p[prev_state][curr_state] * emit_p[curr_state][obs[t]]
                except KeyError:  # the transition/emission doesnt exist
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


def corpus_reader(param):
    pattern = '((?<=\(\')[^\']*(?=\',)|(?<=\(\")[^\"]*(?=\",))'
    sentences = []
    with open( param ) as f:
        corpus = f.readlines()
    for line in corpus:
        temp = re.findall( pattern, line )
        sentences.append( temp )
    ex=sentences[180]
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
    for itr in range( len( temp2 ) ):
        init_prob[temp1[itr]] = float( temp2[itr] )

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
    for line in corpus:
        temp = line.split( ' ' )
        ini = temp[0]
        end = temp[2]
        temp = temp[4].split( '\n' )
        num = float( temp[0] )
        if ini == "Begin":
            continue
        if ini not in trans_p.keys():
            trans_p[ini] = {end: num}
        else:
            trans_p[ini].update( {end: num} )

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
        end = pair[0]
        start = pair[1]
        num = float( temp2[itr] )

        if start not in emi_p.keys():
            emi_p[start] = {end: num}
        else:
            emi_p[start].update( {end: num} )

    return emi_p


if __name__ == "__main__":
    start_time = time.time()
    observed_sentences = corpus_reader( "./corpus.txt" )
    initial_prob_dict = initial_prob_reader( "./transition.txt" )
    states = state_reader( "./temp.txt" )
    transitions = transitions_reader( "./transition.txt" )
    emission_prob = emission_reader( './emission.txt' )

    output_sequences = []
    # Todo everything works except the sentence an number 1855 in the corpus, its too big for me to analyse to find
    #  the error, also i changed \\/ to \/ in the corpus, Also must normalize the emission and transition
    #  probabilities and maybe also initial probablities

    for sent in observed_sentences:
        output = viterbi( sent, states, initial_prob_dict, transitions, emission_prob )
        output_sequences.append( output )

    with open( 'outfile.txt', 'w' ) as file:
        file.writelines( '  '.join( str( j ) for j in i ) + '\n' for i in output_sequences )
    print( "--- %s seconds ---" % (time.time() - start_time) )
