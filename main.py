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
    ontarget_file_path = f"{job_path}/ontarget_{job_name}.txt"

    if verbose: print(f"{job_name} Start!!")
    data = method(
        num_set = num_set,
        threshold= threshold,
        log = True,
        job_name = job_name,
        save_path = path,
    )
    if verbose: print(f"Finish generate on-target.\nsave path = {job_path}")
    if verbose: print("Generate cas-offinder script")
    make_cas_offinder_script(
        query_seq_list = data.ontarget_data,
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
        query_text_path  = ontarget_file_path,
        job_name         = job_name,
        save_csv         = True,
        max_mismatch     = 4,
        slice_length_list= [7,7,6],
    )
    if verbose: print(f'{job_name} Finished')
    return result_df

# Get class by name
class double_mismatch_method:
    @classmethod
    def return_method_list(cls,):
        module_name = 'seq_generator.data_generator'
        module_info = pyclbr.readmodule(module_name)
        method_list = list(module_info.keys())
        return method_list

    @classmethod
    def return_method_class(cls,method='generate_12_data'):
        method_list = cls.return_method_list()
        if method in method_list:
            _class = getattr(data_generator,method)
        else:
            assert False, f"check method, {method}, {method_list}"
        return _class
    

def main():
    parser = argparse.ArgumentParser(description="Q-seq-maker tutorial")

    #Arguments
    parser.add_argument(
        "--name",type=str,default=None,
        help="Job name")
    parser.add_argument(
        "--num_set","-n", type=int, required=True,
        help="Number of sequence set")
    parser.add_argument(
        "--threshold","-t", type=int, default=7,
        help="Edit distance threshold")
    parser.add_argument(
        "--path","-p", type=str, default='./data',
        help="Path to save data")
    parser.add_argument(
        "--method","-m", type=str, default=generate_12_data.__name__,
        help=f"Method of generating on-target sequence\n{double_mismatch_method.return_method_list()}")
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

    if args.name:
        _job_name = f"{args.name}_"
    else:
        _job_name = ''

    num_set   = args.num_set
    threshold = args.threshold
    method    = double_mismatch_method.return_method_class(args.method)
    job_name  = f"{_job_name}{num_set}_{method.__name__}" 
    path      = args.path

    query_seq_generate(
            num_set = num_set,
            threshold= threshold,
            job_name= job_name,
            path = path,
            method= method,
            verbose=verbose,
        )




if __name__=="__main__":
    main()
    #num_set = 2000
    #query_seq_generate(
    #    num_set = num_set,
    #    threshold= 6,
    #    job_name= f"generate_{num_set}_set",
    #    path = './data',
    #    method= generate_12_data,
    #    verbose=True
    #)
