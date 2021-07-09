import numpy as np
import editdistance as edit

# Divide pass_sequence and discard_sequence
def divide_seq_edit_distance(safe_seq_array, add_seq_array, threshold=7):
    '''
    input:
        safe_seq_array: seq list that need not be calculate distance.
                        if do not have seq list, insert None or []
        add_seq_array : seq list that have to check distance.
        threshold     : edit distnace threshold
    output:
        pass_seq      : seq list that farther than threshold
        dis_seq       : discard seq list
        dis_matrix    : distance matrix
        
    기존에 가지고 있던 seq list에 대해서는 edit distance를 구하지 않음
    새로 추가된 seq에 대해서만 distance를 구해서 기존 것과 새로운 것들에 대해서 검증 함
    이미 내가 가진 seq list의 데이터가 많은 경우 시간 단축이 많이 됨
    '''
    import editdistance as edis
    assert type(threshold) in [int, float], f"threshold : {threshold}, {type(threshold)}"
    # check if no safe_seq
    if isinstance(safe_seq_array, (list, np.ndarray)):
        if len(safe_seq_array) == 0:
            _seq_array = add_seq_array
        else:
            _seq_array = np.append(safe_seq_array,add_seq_array, axis=0)
    else:
        _seq_array = add_seq_array
        safe_seq_array = []
    _seq_array = np.array(_seq_array)
    all_num = len(_seq_array)
    safe_num = len(safe_seq_array)
    add_num = len(add_seq_array)
    
    # make distance matrix, if itself or need not calculate value is -1
    #  [-1, -1, 11, 13, 10],
    #  [-1, -1, 12, 14, 13],
    #  [11, 12, -1, 10, 13],
    #  [13, 14, 10, -1, 12],
    #  [10, 13, 13, 12, -1] 
    dis_matrix = -np.ones((all_num,all_num),dtype=np.int8)
    for i in range(all_num-1,safe_num-1,-1):
        for j in range(i,0-1,-1):
            if i != j:
                _dis = np.int8(edis.eval(_seq_array[i],_seq_array[j]))
                dis_matrix[i,j] = _dis
                dis_matrix[j,i] = _dis
                
    # check less than threshold
    dis_index_list_bool = np.where(np.logical_and(dis_matrix>=0,dis_matrix<threshold))
    dis_index_list = []
    for _index_1, _index_2 in zip(*dis_index_list_bool):
        if _index_1 in dis_index_list or _index_2 in dis_index_list:
            continue
        max_index = max(_index_1,_index_2)
        min_index = min(_index_1,_index_2)
        if min_index < safe_num:
            dis_index_list.append(max_index)
            continue
        _index_1_sum = np.sum(dis_matrix[_index_1])
        _index_2_sum = np.sum(dis_matrix[_index_2])
        if _index_1_sum > _index_2_sum:
            dis_index_list.append(_index_2)
        else:
            dis_index_list.append(_index_1)
    pass_index_list = np.delete(np.arange(all_num),dis_index_list).astype(np.int)
    dis_index_list = np.array(dis_index_list, dtype=np.int)
    
    pass_seq = _seq_array[pass_index_list]
    dis_seq = _seq_array[dis_index_list]
        
    return pass_seq, dis_seq, dis_matrix
            


def eval_dis_matrix(seq_array):
    import editdistance as edis
    num = len(seq_array)
    dis_matrix = np.zeros((num, num))
    for i in range(num):
        for j in range(i,num):
            if i != j:
                _dis = edis.eval(seq_array[i],seq_array[j])
                dis_matrix[i,j] = _dis
                dis_matrix[j,i] = _dis
            else:
                dis_matrix[i,j] = 0
    return dis_matrix

def classify_seq(seq_array, threshold=10):
    '''
    It has a problem that does not detect same sequence
    '''
    assert type(threshold) in [int, float], f"threshold : {threshold}, {type(threshold)}"
    num = len(seq_array)
    dis_matrix = eval_dis_matrix(seq_array)
    dis_index_list_bool = np.where(np.logical_and(dis_matrix>0, dis_matrix<threshold))
    dis_index_list = []
    for _index_1, _index_2 in zip(dis_index_list_bool[0],dis_index_list_bool[1]):
        if _index_1 in dis_index_list or _index_2 in dis_index_list:
            continue
        _index_1_sum = np.sum(dis_matrix[_index_1])
        _index_2_sum = np.sum(dis_matrix[_index_2])
        if _index_1_sum > _index_2_sum:
            dis_index_list.append(_index_2)
        else:
            dis_index_list.append(_index_1)
    pass_index_list = np.delete(np.arange(num),dis_index_list).astype(np.int)
    dis_index_list = np.array(dis_index_list,dtype=np.int)
    
    pass_seq = seq_array[pass_index_list]
    discard_seq = seq_array[dis_index_list]
    return pass_seq, discard_seq
