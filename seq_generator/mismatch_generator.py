from encoder_decoder.seq_encoder_decoder import *
from seq_generator.query_generator import *
from seq_evaluator.evaluator import *

import numpy as np
import editdistance as edis


# nC2 method
# generate 2 mismatch sequence
def double_nC2_seq_list(seq, first_change=True):
    '''
    input : 
        seq : Query sequence
        first_change : Change position
    Select all position in sequence(nC2)
    Convert to the others nucleotides at one position(first or seconde, Args first_change)
    output : [[ACGT,CGT,GGT,...],[...]]
    '''
    assert not isinstance(seq[0], (int,float,complex))
    if type(seq) == str:
        seq = list(seq)
    seq_num = acgt2num(seq)
    acgt_num_list = np.arange(1,5)
    mis2_seq_list = []
    for pos_1 in range(len(seq_num)):
        for pos_2 in range(pos_1+1,len(seq_num)):
            if first_change:
                _change_point = pos_1
                _fix_point    = pos_2
            else:
                _change_point = pos_2
                _fix_point    = pos_1
            _change_point_NT = seq_num[_change_point]
            _fix_point_NT    = seq_num[_fix_point]
            _fix_NT_change_list = np.delete(acgt_num_list,_fix_point_NT-1)
            _fix_point_m_NT = _fix_NT_change_list[np.random.randint(len(_fix_NT_change_list))]
            for _change_point_m_NT in np.delete(acgt_num_list,_change_point_NT -1):
                _tmp_seq_num = np.array(seq_num).copy()
                _tmp_seq_num[_change_point] = _change_point_m_NT
                _tmp_seq_num[_fix_point]    = _fix_point_m_NT
                mis2_seq_list.append(num2acgt(_tmp_seq_num))
    return np.array(mis2_seq_list)


def distance_based_change_diff_length_list(seq, length, num):
    '''
    input : 
        seq: query sequence
        length: length of distance to change position
        num: number of generate mismatch sequence
    Generate double mismatch sequence based distance to change.
    Only use when bar length is less than query squence length
    output : [ACGT,AGGC,...]
    '''
    assert not isinstance(seq[0], (int,float,complex))
    assert len(seq) > length, f"Length has to less than seq length, length:{length}"
    assert len(seq) >= length + num -1, f"Request too many numbers of seq, max num : {len(seq)-length+1}"
    seq_num = acgt2num(seq)
    change_rule_dict = {
        1 : [2,3], #A -> C,G
        2 : [1,4], #C -> A,T
        3 : [1,4], #G -> A,T
        4 : [2,3], #T -> C,G
        5 : [1,2,3,4],
    }
    bar_length = length
    choose_num = num
    seq_length = len(seq_num)
    avail_position_list = np.arange(0,seq_length-bar_length+1)
    mutate_position_list =np.sort( np.random.choice(avail_position_list,choose_num,replace=False))
    mis2_seq_list = []
    for _pos in mutate_position_list:
        _tmp_seq = np.array(seq_num).copy()
        for i in [_pos,_pos+bar_length-1]:
            _pos_NT = seq_num[i]
            _m_NT_list = change_rule_dict.get(_pos_NT,[1,2,3,4])
            _tmp_seq[i] = _m_NT_list[np.random.randint(len(_m_NT_list))]
        mis2_seq_list.append(num2acgt(_tmp_seq))
    return np.array(mis2_seq_list)


def distance_based_change_same_length_list(seq,num):
    '''
    input : 
        seq: query sequence
        num: number of generate mismatch sequence
    Only use when bar length is equal to query sequence length
    output : [ACGT,CCGT,..]
    '''
    assert not isinstance(seq[0], (int,float,complex))
    assert num <= 9, f"Request too many number of seq, max : 9"
    seq_num = acgt2num(seq)
    change_rule_dict = {
        1 : [2,3], #A -> C,G
        2 : [1,4], #C -> A,T
        3 : [1,4], #G -> A,T
        4 : [2,3], #T -> C,G
        5 : [1,2,3,4],
    }
    acgt_list = np.arange(1,4+1)
    bar_length = len(seq_num)
    choose_num = num
    seq_length = len(seq_num)
    mis2_seq_list = []
    
    _pos_1_NT = seq_num[0]
    _pos_2_NT = seq_num[-1]
    for _mut_1 in np.delete(acgt_list,_pos_1_NT -1):
        for _mut_2 in np.delete(acgt_list,_pos_2_NT -1):
            _tmp_seq = np.array(seq_num).copy()
            _tmp_seq[0]  = _mut_1
            _tmp_seq[-1] = _mut_2
            mis2_seq_list.append(num2acgt(_tmp_seq))
    mis2_seq_list = np.random.choice(np.array(mis2_seq_list),num,replace=False)
    return mis2_seq_list

def distance_based_change_list(seq, length, num):
    '''
    input : 
        seq:  query sequence
        length: length of distance to change position
        num : number of generate mismatch sequence
    Merge the two case of different lengths and equal lengths
    output : [ACGT,ACGG,...]
    '''
    assert not isinstance(seq[0], (int,float,complex))
    assert len(seq) >= length, f"Check length, seq:{len(seq)}, length:{length}"
    
    if len(seq) == length:
        return distance_based_change_same_length_list(seq,num)
    else:
        return distance_based_change_diff_length_list(seq,length,num)


def generate_distance_based_seq_list(seq, pair_list=None):
    '''
    input :
        seq: query sequence
        pair_list: (distance, number of seq) pair list, if None take default
    output : [ACGT, ACCG, ....]
    '''
    if pair_list == None:
        pair_list = []
        pair_2 = (2,10)
        pair_list.append(pair_2)
        for i in range(3,5+1):
            pair_list.append((i,8))
        for i in range(6,9+1):
            pair_list.append((i,6))
        for i in range(10,20+1):
            pair_list.append((i,2))
    return np.concatenate(list(map(lambda x : distance_based_change_list(seq,*x),pair_list)))


# generate one mismatch seq
def one_mismatch_seq_list(seq):
    '''
    input :
        seq: query sequence
    Generate one mismatch sequence
    output : [ACGT,CCGT,...]
    '''
    if type(seq) == str:
        seq = list(seq)
    seq_num = acgt2num(seq)
    acgt_num_list = np.arange(1,5)
    mis1_seq_list = []
    for pos in range(len(seq_num)):
        _pos_NT = seq_num[pos]
        for _change_point_m_NT in np.delete(acgt_num_list,_pos_NT-1):
            _tmp_seq = np.array(seq_num).copy()
            _tmp_seq[pos] = _change_point_m_NT
            mis1_seq_list.append(num2acgt(_tmp_seq))
    mis1_seq_list = np.array(mis1_seq_list)
    return mis1_seq_list
