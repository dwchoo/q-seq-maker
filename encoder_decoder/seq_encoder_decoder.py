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
