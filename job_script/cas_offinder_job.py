from seq_generator.data_generator import generate_mismatch_data
from encoder_decoder.seq_encoder_decoder import query_type_code_encoder, gc_at_classify_encoder

import numpy as np
import pandas as pd
from pathlib import Path, PureWindowsPath

def make_cas_offinder_script(
    query_seq_list,
    job_name,
    job_path,
    ref_path='/root/gen_ref/hg38.fa',
    PAM='NGG',
    check_mismatch=4,
):
    '''
    args:
        query_seq_list : query_list, [ACGT,AACG,...]
        job_name       : job_name(it is same of folder name), ./job_name/job_name_script.sh
        job_path       : job files path, ./job_name
        ref_path       : reference file(hg38, hbg19..) path
        PAM            : PAM, NGG
        check_mismatch : number of mismatch cas-offinder will find
    output:
        cas_offinder runing script, input file that contain query sequence
    '''
    query_seq_list = np.array(query_seq_list)
    if PureWindowsPath(job_path).suffix:
        file_path = Path(job_path).parent
    else:
        file_path = f"{job_path}"
    Path(file_path).mkdir(parents=True, exist_ok=True)

    # make cas-offinder script
    with open(f"{file_path}/run_{job_name}_script.sh",'w') as file:
        file.write(f"#!/bin/bash\n")
        file.write(f"""command=`cas-offinder $(dirname "$0")/{job_name}_input.txt G $(dirname "$0")/{job_name}_output.txt`\n""")
        file.write(f"echo $command")

    # make input file
    with open(f"{file_path}/{job_name}_input.txt", 'w') as file:
        file.write(f"{ref_path}\n")
        file.write(f"NNNNNNNNNNNNNNNNNNNN{PAM}\n")
        assert len(query_seq_list.shape) == 1, f"Check query shape: {query_seq_list.shape}"
        for _seq in query_seq_list:
            file.write(f"{_seq}NNN {check_mismatch}\n")
    

def analysis_cas_offinder_result(
    output_file_path,
    query_text_path,
    save_csv=True,
    max_mismatch = 4,
    slice_length_list = [7,7,6],
):
    '''
    args:
        output_file_path  : cas-offinder output file path,
                            ./{job_name}/{job_name}_output.txt
        query_text_path   : query text file path,
                            ./{job_name}/{job_name}_query.txt
        save_csv          : save result by csv file,
                            it is saved in the  same path as the output file
        max_mismatch      : the maximum value of number of mismatch
        slice_length_list : slice point, [7,7,6]
    output:
        DataFrame, and csv file
    '''
    # load cas-offinder output file
    header = ['query','chr','site','seq','direction','mismatch']
    data_type = {
        'query'    : 'category',
        'chr'      : 'category',
        'site'     : 'uint32',
        'mismatch' : 'uint8',
    }
    data_frame = pd.read_csv(
        output_file_path,
        sep='\t',
        names=header,
        dtype=data_type,
    )
    # load query sequence text file
    query_list = np.array(generate_mismatch_data.read_sequence_tolist(query_text_path))
    # convert sequence to type code
    query_type_code_list = query_type_code_encoder.batch_seq_type_code_encoder(
        seq_list= query_list,
        slice_length_list= slice_length_list,
        only_code= True
    )

    # analysis number of mismatch in each sequence.
    query_set = data_frame['query'].unique()
    query_mismatch_info = {}
    for _query in query_set:
        query_mismatch_info[_query] = \
            data_frame[data_frame['query'] == _query]['mismatch'].value_counts().to_dict()
    
    # Dictionary -> list
    query_mismatch_info_list = []
    for _query in query_list.reshape(-1):
        _tmp_list = []
        _tmp_query_info = query_mismatch_info.get(f'{_query}NNN',{})
        for i in range(max_mismatch+1):
            _tmp_list.append(_tmp_query_info.get(i,0))
        query_mismatch_info_list.append(_tmp_list)
    query_mismatch_info_list = np.array(query_mismatch_info_list)
    # calculate cg - at ratio
    query_cg_at_ratio_list = gc_at_classify_encoder.calc_batch_seq_gc_at_ratio(query_set)

    # concatenate all of data
    analysis_result_list = np.concatenate(
        [np.reshape(query_type_code_list,(-1,1)),
        np.reshape(query_list,(-1,1)),
        query_cg_at_ratio_list,
        query_mismatch_info_list],
        axis=1
    )
    columns_header = [
            'Type_code',
            'query',
            'GC_rate',
            'AT_rate',
            #'mis_0','mis_1','mis_2','mis_3','mis_4'
    ] + [f"mis_{i}" for i in range(max_mismatch+1)]
    # make DataFrame
    query_info_df = pd.DataFrame(
        analysis_result_list,
        columns=columns_header,
    )

    # save csv file
    if save_csv:
        file_path = Path(output_file_path).parent
        query_info_df.to_csv(f'{file_path}/query_analysis.csv',index=False)

    return query_info_df
    
