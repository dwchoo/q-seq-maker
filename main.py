from seq_generator.query_generator import *
from seq_generator.mismatch_generator import *
from seq_generator.data_generator import *
from job_script.cas_offinder_job import make_cas_offinder_script, run_cas_offinder, analysis_cas_offinder_result
from job_script.cas_offinder_job import *

import numpy as np



def query_seq_generate(
    num_set,
    threshold = 7,
    job_name='generate_nC2',
    path = './',
    method=generate_8_nC2_data,
    verbose=False,
):
    job_path = f"{path}/{job_name}"
    cas_offinder_script_path = f"{job_path}/run_{job_name}_script.sh"
    output_file_path = f"{job_path}/{job_name}_output.txt"
    query_file_path = f"{job_path}/query_{job_name}.txt"

    if verbose: print("Job Start!!")
    data = method(
        num_set = num_set,
        threshold= threshold,
        log = True,
        job_name = job_name,
        save_path = path,
    )
    if verbose: print("Finish generate query.\nGenerate cas-offinder script")
    make_cas_offinder_script(
        query_seq_list = data.query_data,
        job_name = job_name,
        job_path = job_path,
    )
    if verbose: print("Start to run cas-offinder")
    run_cas_offinder(
        file_path = cas_offinder_script_path,
        log = True,
    )
    if verbose: print("Finish cas-offinder.\nStart to analysis results")
    result_df = analysis_cas_offinder_result(
        output_file_path = output_file_path,
        query_text_path  = query_file_path,
        job_name         = job_name,
        save_csv         = True,
        max_mismatch     = 4,
        slice_length_list= [7,7,6],
    )
    if verbose: print('Job Finished')
    return result_df




if __name__=="__main__":
    num_set = 10
    query_seq_generate(
        num_set = num_set,
        threshold= 7,
        job_name= f"generate_{num_set}_set",
        path = './data',
        method= generate_8_nC2_data,
        verbose=True
    )
