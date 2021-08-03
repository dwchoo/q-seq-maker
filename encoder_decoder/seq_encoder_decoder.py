from __future__ import absolute_import, unicode_literals
import numpy as np


class one_hot_encoder:
    one_hot_encoder_dict = {
        'A' : np.array([1,0,0,0],dtype=np.int8),
        'a' : np.array([1,0,0,0],dtype=np.int8),
        'C' : np.array([0,1,0,0],dtype=np.int8),
        'c' : np.array([0,1,0,0],dtype=np.int8),
        'G' : np.array([0,0,1,0],dtype=np.int8),
        'g' : np.array([0,0,1,0],dtype=np.int8),
        'T' : np.array([0,0,0,1],dtype=np.int8),
        't' : np.array([0,0,0,1],dtype=np.int8),
        
        'R' : np.array([1,0,1,0],dtype=np.int8),
        'Y' : np.array([0,1,0,1],dtype=np.int8),
        'S' : np.array([0,1,1,0],dtype=np.int8),
        'W' : np.array([1,0,0,1],dtype=np.int8),
        'K' : np.array([0,0,1,1],dtype=np.int8),
        'M' : np.array([1,1,0,0],dtype=np.int8),
        
        'B' : np.array([0,1,1,1],dtype=np.int8),
        'D' : np.array([1,0,1,1],dtype=np.int8),
        'H' : np.array([1,1,0,1],dtype=np.int8),
        'V' : np.array([1,1,1,0],dtype=np.int8),
        
        'N' : np.array([1,1,1,1],dtype=np.int8),
        'n' : np.array([1,1,1,1],dtype=np.int8), #'n'????
    }
    
    @classmethod
    def letter_encoder(cls, letter):
        return cls.one_hot_encoder_dict[letter]
    
    @classmethod
    def seq_encoder(cls, sequence):
        one_hot_seq = list(map(cls.letter_encoder, sequence))
        one_hot_seq = np.array(one_hot_seq)
        return one_hot_seq
    
    @classmethod
    def batch_seq_encoder(cls, batch_seq):
        batch_seq_one_hot = []
        for seq in batch_seq:
            batch_seq_one_hot.append(cls.seq_encoder(seq))
        return np.array(batch_seq_one_hot)
    
    @classmethod
    def average_distribution(cls, seq_list):
        seq_num     = len(seq_list)
        seq_length  = len(seq_list[0])

        dist_arr = np.zeros((seq_length, 4))
        for seq in seq_list:
            dist_arr += cls.seq_encoder(seq)
        return dist_arr.T/seq_num
    
    
class one_hot_decoder:
    one_hot_decoder_dict = {
        '1000' : 'A',
        '0100' : 'C',
        '0010' : 'G',
        '0001' : 'T',
        
        '1010' : 'R',
        '0101' : 'Y',
        '0110' : 'S',
        '1001' : 'W',
        '0011' : 'K',
        '1100' : 'M',
        
        '0111' : 'B',
        '1011' : 'D',
        '1101' : 'H',
        '1110' : 'V',
        
        '1111' : 'N'
    }
    
    @classmethod
    def letter_decoder(cls, letter_code):
        return cls.one_hot_decoder_dict[letter_code]
    
    @classmethod
    def seq_decoder(cls, seq_one_hot):
        seq_letter_list = []
        for letter_one_hot in seq_one_hot:
            one_hot_pos_list = np.array(letter_one_hot, dtype=np.int8)
            one_hot_pos_code = ''.join(map(str,one_hot_pos_list))
            seq_letter_list += cls.letter_decoder(one_hot_pos_code)
        seq = ''.join(seq_letter_list)
        return seq


class mismatch_calculator:
    prime_convert_main = np.array([1,2,3,5], dtype=np.int8)
    prime_convert_sub  = np.array([7,11,13,17], dtype=np.int8)

    prime_main_encoder_dict = {
        'A' : 1,
        'a' : 1,
        'C' : 2,
        'c' : 2,
        'G' : 3,
        'g' : 3,
        'T' : 5,
        't' : 5,
        
        'N' : 0,
        'n' : 0, #'n'????
    }

    prime_sub_encoder_dict = {
        'A' : 7,
        'a' : 7,
        'C' : 11,
        'c' : 11,
        'G' : 13,
        'g' : 13,
        'T' : 17,
        't' : 17,
        
        'N' : 0,
        'n' : 0, #'n'????
    }
    
    prime_multi_encoder_dict = {
        7  : None,
        11 : 'A>C',
        13 : 'A>G',
        17 : 'A>T',
        14 : 'C>A',
        22 : None,
        26 : 'C>G',
        34 : 'C>T',
        21 : 'G>A',
        33 : 'G>C',
        39 : None,
        51 : 'G>T',
        35 : 'T>A',
        55 : 'T>C',
        65 : 'T>G',
        85 : None,
        
        0  : None, # N,n
    }

    @classmethod
    def seq_prime_encoder(cls, query, type_dict):
        prime_encoded_query = list(map(type_dict.get,query))
        prime_encoded_query = np.array(prime_encoded_query,dtype=np.int8)
        return prime_encoded_query

    @classmethod
    def __seq_mismatch_calculator(cls, query_1, query_2):
        query_1_prime_encoded = cls.seq_prime_encoder(query_1, cls.prime_main_encoder_dict)
        query_2_prime_encoded = cls.seq_prime_encoder(query_2, cls.prime_sub_encoder_dict)

        query_multiply = query_1_prime_encoded * query_2_prime_encoded
        mismatch_list = list(map(cls.prime_multi_encoder_dict.get,query_multiply))

        mismatch_info_list = []
        for index, _info in enumerate(mismatch_list):
            if _info:
                _info_str = f"{index}:{_info}"
                mismatch_info_list.append(_info_str)
        return mismatch_info

    @classmethod
    def seq_mismatch_info_list(cls, query_1, query_2):
        mismatch_info_list = cls.__seq_mismatch_calculator(query_1, query_2)
        return mismatch_info_list

    @classmethod
    def seq_mismatch_info_str(cls, query_1, query_2):
        mismatch_info_list = cls.__seq_mismatch_calculator(query_1, query_2)
        mismatch_info_str = ','.join(mismatch_info_list)
        return mismatch_info_str






# ACCCGT -> 122234
def acgt2num(seq):
    '''
    input : ACGT
    Convert sequence to number
    output : 1234
    '''
    acgt_dict = {
        'A' : 1,
        'C' : 2,
        'G' : 3,
        'T' : 4,
    }
    return list(map(lambda x : acgt_dict.get(x,None),seq))


# 1223 -> ACCGT
def num2acgt(seq_num,join=True):
    '''
    input : 1234
    Convert number to sequence
    output : ACGT
    '''
    num_acgt_dict = {
        1 : 'A',
        2 : 'C',
        3 : 'G',
        4 : 'T',
        5 : 'N'
    }
    result = list(map(lambda x : num_acgt_dict.get(x,'-'),seq_num)) 
    if join:
        result = ''.join(result)
    else:
        result = np.array(result)
    return result


# calculate GC <-> AT ratio
class gc_at_classify_encoder:
    classifier = {
        'G' : np.array([1,0], dtype=np.int8),
        'C' : np.array([1,0], dtype=np.int8),
        'A' : np.array([0,1], dtype=np.int8),
        'T' : np.array([0,1], dtype=np.int8),
    }
    
    @classmethod
    def letter_encoder(cls, letter):
        return cls.classifier.get(letter,np.array([0,0],dtype=np.int8))
    
    @classmethod
    def seq_encoder(cls, sequence):
        classified_seq = list(map(cls.letter_encoder,sequence))
        classified_seq = np.array(classified_seq)
        return classified_seq
    
    @classmethod
    def calc_seq_gc_at_ratio(cls, sequence):
        classified_seq = cls.seq_encoder(sequence)
        sum_column = np.sum(classified_seq, axis=0)
        sum_all = np.sum(classified_seq)
        return sum_column / sum_all
    
    @classmethod
    def batch_seq_encoder(cls, batch_seq):
        batch_seq_classified_list = []
        for seq in batch_seq:
            batch_seq_classified_list.append(cls.seq_encoder(seq))
        return np.array(batch_seq_classified_list)
    
    @classmethod
    def calc_batch_seq_gc_at_ratio(cls, batch_seq):
        batch_seq_classified_list = []
        for seq in batch_seq:
            batch_seq_classified_list.append(cls.calc_seq_gc_at_ratio(seq))
        return np.array(batch_seq_classified_list)



class query_type_code_encoder:

    @classmethod
    def seq_type_code_encoder(cls, seq, slice_length_list=[7,7,6]):
        '''
        arguments:
            seq: query sequence, ACGTAAA
            slice_length_list: [7,7,6]
        return:
            type_code, 123
        '''
        seq_encoded = gc_at_classify_encoder.seq_encoder(seq)
        slice_position = cls.__make_slice_position(cls, slice_length_list)
        type_code_list = []
        for _start, _end in slice_position:
            _seq_block = seq_encoded[_start:_end]
            _type_code = cls.return_code(cls, _seq_block)
            type_code_list.append(str(_type_code))
        type_code = ''.join(type_code_list)
        return type_code

    @classmethod
    def batch_seq_type_code_encoder(cls, seq_list, slice_length_list=[7,7,6],only_code=True):
        seq_array = np.array(seq_list)
        assert len(seq_array.shape) == 1, \
                f"Check seq_list shape, {seq_array.shape},{seq_array.dtype}"
        batch_type_code_seq_list = []
        batch_type_code_list = []
        if only_code:
            for _seq in seq_array:
                _type_code = cls.seq_type_code_encoder(_seq, slice_length_list)
                batch_type_code_list.append(_type_code)
            return np.array(batch_type_code_list)

        else:
            for _seq in seq_array:
                _type_code = cls.seq_type_code_encoder(_seq, slice_length_list)
                batch_type_code_seq_list.append((_type_code, _seq))
            return batch_type_code_seq_list
        


    def __make_slice_position(self, slice_length_list):
        '''
        input: slice length
            [7,7,6]
        output: slice position
            [(0,7),(7,14),(14,20)]
        '''
        slice_position = []
        previous_position = 0
        next_position = 0
        for _length in slice_length_list:
            _start = previous_position
            _end = _start + _length
            slice_position.append((_start,_end))
            previous_position = _end
        return slice_position

    def return_code(self, seq_block):
        '''
        input: seq_block
            [0, 1],
            [0, 1],
            [0, 1],
        output: 1
            1: GC > AT
            2: GC < AT
            3: CG = AT
        '''
        gc_at_sum = np.sum(seq_block, axis=0)
        assert len(gc_at_sum) == 2, f"Check seq_block shape, {seq_block.shape}"
        if np.greater(*gc_at_sum):
            return 1
        elif np.less(*gc_at_sum):
            return 2
        elif np.equal(*gc_at_sum):
            return 3
        else:
            return None
