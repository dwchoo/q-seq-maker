#!/usr/bin/env python
# -*- coding: utf-8 -*-

from seq_generator.query_generator import *
from seq_generator.mismatch_generator import *
from seq_generator.data_generator import *
from seq_generator import data_generator
from job_script.cas_offinder_job import make_cas_offinder_script, run_cas_offinder, analysis_cas_offinder_result
from job_script.cas_offinder_job import *

import numpy as np
import argparse
import pyclbr
import sys
from pathlib import Path


def ontarget_file_analyser(
    ontarget_text_file_path,
    job_name='generate_nC2',
    save_path = './',
    PAM='NGG',
    PAM_end=True,
    verbose=False,
):
    job_path = f"{save_path}/{job_name}"
    cas_offinder_script_path = f"{job_path}/run_{job_name}_script.sh"
    output_file_path = f"{job_path}/{job_name}_output.txt"

    if Path(ontarget_text_file_path).is_file():
        data_list = data_generator.generate_mismatch_data.read_sequence_tolist(
            file_name=ontarget_text_file_path,
        )
    else:
        sys.exit(f"Cannot find file. Please check path again.\npath:{ontarget_text_file_path}")


    if verbose: print(f"{job_name} Start!!")
    if verbose: print("Generate cas-offinder script")
    # Sequence file have comment and lower character
    data_list = data_generator.seq_processing.delete_comment_in_list(
        data_list = data_list,
        letter='#'
    )
    data_list = data_generator.seq_processing.slice_seq_in_list(
        data_list= data_list,
        max_length= 20,
        cut_end = True,
    )
    make_cas_offinder_script(
        query_seq_list = data_list,
        job_name = job_name,
        job_path = job_path,
        PAM      = PAM,
        PAM_end  = PAM_end,
    )
    if verbose: print("Start to run cas-offinder")
    run_cas_offinder(
        job_name  = job_name,
        file_path = cas_offinder_script_path,
        log = True,
        log_head_message=f"PAM:{PAM}, PAM_end:{PAM_end}\
        \nOntarget file path: {ontarget_text_file_path}",
    )
    if verbose: print("Finish cas-offinder.\nStart to analysis results")
    result_df = analysis_cas_offinder_result(
        output_file_path = output_file_path,
        query_list       = data_list,
        job_name         = job_name,
        save_csv         = True,
        PAM              = PAM,
        PAM_end          = PAM_end,
        max_mismatch     = 4,
        slice_length_list= [7,7,6],
    )
    if verbose: print(f'{job_name} Finished')
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Q-seq-maker analyser tutorial")

    #Arguments
    parser.add_argument(
        "--source","-s", type=str, required=True,
        help="Ontarget sequence list file path")
    parser.add_argument(
        "--name",type=str,default=None,
        help="Job name")
    parser.add_argument(
        "--PAM","-P", type=str, default='NGG',
        help="PAM, 'NGG','TTTV'...")
    parser.add_argument(
        "--PAM_end","-E", type=int, default=1,
        help="PAM end True=1 False=0, CAS9 1, Cpf1 0")
    parser.add_argument(
        "--path","-p", type=str, default='./data',
        help="Path to save data")
    parser.add_argument(
        "--verbose",'-v', action="store_true", default=True,
        help="Verbose")
    
    # Check no arguments
    if len(sys.argv)==1:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    if args.verbose:
        verbose = True
    else:
        verbose = False

    ontarget_text_file_path = args.source
    PAM       = args.PAM
    PAM_end   = False if int(args.PAM_end) == 0 else True
    job_name  = '_'.join(filter(None,[args.name, f"analysis_ontarget"]))
    path      = args.path

    ontarget_file_analyser(
            ontarget_text_file_path= ontarget_text_file_path,
            job_name= job_name,
            save_path = path,
            PAM = PAM,
            PAM_end = PAM_end,
            verbose=verbose,
        )




if __name__=="__main__":
    main()
