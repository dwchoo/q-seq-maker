#from encoder_decoder.seq_encoder_decoder import *
#from ..encoder_decoder import seq_encoder_decoder

import numpy as np
import editdistance as edis


# random sequence generator
def random_seq(length=20, column=4):
    return np.random.randint(column, size=length)


# generate condition sequence

# genearte list of seq [A,C,G,T,T...]
def generate_ontarget_seq_set_list(gen_num_each=1, concate=False,equal=True,join=True):
    '''
    input :
        gen_num_each : number of sequence set
        concate      : if set is more than 2, [[ACGT,ACCG,..], [...]]
        equal        : equal type
        join         : [A,C,G,T] -> [ACGT]
    output : [ACGT,CGGT, ...] or [[ACGT,CCA,..],[...],...]
    '''
    _seq_type_list = generate_seq_type_list(equal)
    ontarget_list = []
    for _seq_type in _seq_type_list:
        _tmp_list = []
        for i in range(gen_num_each):
            _tmp_list.append(convert_type2seq(_seq_type,join))
        ontarget_list.append(_tmp_list)
    if concate:
        result = np.concatenate(ontarget_list)
    else:
        result = np.array(ontarget_list)
    return result


# generate block cg=1,at=2 =>['CG','AT','AT'] and shuffle
def block_arrange(cg_num, at_num):
    '''
    input : 3,4
    Generate shuffled CG/AT block
    output : [CG,AT,AT,AT,CG,AT,CG]
    '''
    assert isinstance(cg_num, (int,float,complex)), isinstance(at_num, (int,float,complex))
    cg_at_list = ['CG'] * cg_num + ['AT'] * at_num
    np.random.shuffle(cg_at_list)
    return cg_at_list


# generate seq ['AT','CG','AT'...]
def seq_arrange(seq_type):
    '''
    input : 12e3 (Sequence type)
    Convert sequence type code to sequence
    output : [CG,AT,AT,AT,CG,AT,CG]
    '''
    assert isinstance(seq_type, (list,np.ndarray))
    cg_7_dominate = [(3,4),(2,5),(1,6),(0,7)]
    at_7_dominate = [(4,3),(5,2),(6,1),(7,0)]
    cg_6_dominate = [(2,4),(1,5),(0,6)]
    at_6_dominate = [(4,2),(5,1),(6,0)]
    equal_33  = [(3,3)]
    type_dict = {
        '1'  : cg_7_dominate,
        '2'  : at_7_dominate,
        'e1' : cg_6_dominate,
        'e2' : at_6_dominate,
        'e3' : equal_33,
    }
    seq = []
    for index in seq_type:
        _tmp_type = type_dict[str(index)]
        _letter_num = _tmp_type[np.random.randint(len(_tmp_type))]
        _mass = block_arrange(*_letter_num)
        seq = np.append(seq,_mass)
    return seq


# generate [1,2,'e1']
def generate_seq_types_code_list(equal=True):
    '''
    input : True / False (if last block has equal type)
    Make sequence type code
    output: [1,2,e3]
    '''
    mass_1_type = [1,2]
    mass_2_type = [1,2]
    if equal:
        mass_e_type = ['e1','e2','e3']
    else:
        mass_e_type = ['e1','e2']
    seq_types_list = []
    for pos_1 in mass_1_type:
        for pos_2 in mass_2_type:
            for pos_e in mass_e_type:
                seq_types_list.append([pos_1,pos_2,pos_e])
    return seq_types_list


# seq code 2*2*3 generate
def generate_seq_type_list(equal=True):
    '''
    input : True / False (Last block has equal type or not)
    Generate seq type code all number of cases
    output : [[1,1,e1].[1,1,e2] ... [2,2,e3]]
    '''
    _seq_code_list = generate_seq_types_code_list(equal)
    seq_types_list = []
    for _seq_type in _seq_code_list:
        seq_types_list.append(seq_arrange(_seq_type))
    seq_types_list = np.array(seq_types_list)
    return seq_types_list


# ['AT','CG'] -> ['A','G']
def convert_type2seq(seq_type,join=True):
    '''
    input : [AT,CG,CG,AT ...]
    Choose Nucleotide randomly
    AT -> A or T
    CG -> C or G
    output : [A,C,G,A ...]
    '''
    seq = []
    for _pos in seq_type:
        seq.append(_pos[np.random.randint(2)])
    if join:
        seq = ''.join(seq)
    return np.array(seq)




if __name__=='__main__':
    print(f"hello world")
