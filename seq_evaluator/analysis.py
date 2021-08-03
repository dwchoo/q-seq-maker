from encoder_decoder.seq_encoder_decoder import one_hot_encoder


import numpy as np


def calc_mismatch_from_query_list(query_seq_list, mismatch_seq_list):
    '''
    args:
        query_seq_list      : query sequence list, [ACGT,ACGG,...]
        mismatch_seq_list   : mismatch sequence list, [[ACGG,ACGC,..],[ACGT,ACGC,..]]
    return:
        mismatch_resut      : number of mismatch for each query sequence, [[12,10,13,11],[9,10,8,12],..]
    '''
    assert len(query_set_list) == len(mismatch_seq_list). \
            f"Check seq, query:{len(query_set_list)}, mis:{len(mismatch_seq_list)}"
    batch_length = len(query_seq_list)
    mismatch_result = []
    for i in range(batch_length):
        _query_seq = query_seq_list[i]
        _mismatch_seq_list = mismatch_seq_list[i]
        _mismatch_num_pos = calc_mismatch_each_position(_query_seq,_mismatch_seq_list)
        mismatch_result.append(_mismatch_num_pos)
    mismatch_result = np.array(mismatch_result)
    return mismatch_result



def calc_mismatch_each_position(query, mismatch_list):
    '''
    args:
        query           : query_sequence, ACGT
        mismatch_list   : mismatch seq list, [ACGA,ACGC,ACGG,...]
    return:
        mismatch_num_pos: number of mismatch for each position, [12,13,10,15]
    '''
    query_one_hot = one_hot_encoder.seq_encoder(query)
    mismatch_one_hot_list = one_hot_encoder.batch_seq_encoder(mismatch_list)
    mismatch_calc = np.clip(mismatch_one_hot_list - query_one_hot,0,None)
    mismatch_num_pos = np.sum(mismatch_calc, axis=0)
    return mismatch_num_pos

