from seq_generator.query_generator import *
from seq_generator.mismatch_generator import *
from seq_evaluator.evaluator import *

import numpy as np



class generate_mismatch_data:
    '''
    Generate mismatch data
    One mismatch
    nC2 mismatch
    distance based mismatch
    '''
    def generate_query_seq_data(self,
        num_set,
        threshold,
        equal,
        log=True,
        job_name='query_seq',
        save_path = '.'
    ):
        '''
        input:
            num_set    : number of query set to generate
            threshold  : edit distance threshold 
            equal      : True, if last group has equal case
            log        : True, if save log
            job_name   : string, job name
            save_path  : save file path
        return:
            query_set  : [ACGT,AACG ...]
        '''
        job_path = self.__return_job_folder_path_N_make(job_name=job_name, path=save_path)
        save_file_path = f"{job_path}/{job_name}_query.txt"
        log_file_path = f"{job_path}/log/{job_name}.log"
        if log:
            logger = self.__seq_generate_log(log_file_path,'Query')
        else:
            logger = None
            
        if equal:
            num_query = num_set*12
        else:
            num_query = num_set*8
            
        query_set = []
        while 1:
            previous_query_mass = np.array(query_set).copy()
            new_query_mass = generate_query_seq_set_list(1, concate=True, equal=equal,join=True)
            pass_query_mass, discard_query_mass, _ = divide_seq_edit_distance(
                previous_query_mass,
                new_query_mass,
                threshold,
            )
            if len(discard_query_mass) == 0:
                query_set = pass_query_mass.copy()
                if log:
                    logger.info(f"{len(pass_query_mass)}/{num_query} - save file : {save_file_path}")
                    self.__save_sequence(pass_query_mass,save_file_path)
            else:
                query_set = previous_query_mass.copy()
            if len(query_set) >= num_query:
                pass_tmp, discard_tmp, _ = divide_seq_edit_distance(None, query_set, threshold)
                if log:
                    logger.info(f"FINISH pass_num : {len(pass_tmp)}, discard_num : {len(discard_tmp)}")
                assert len(discard_tmp) == 0, f"discard seq num: {len(discard_tmp)}"
                return query_set
    
    def generate_one_mis_data(self, query_list):
        data_one_mis_list = np.array(list(map(one_mismatch_seq_list,query_list)))
        return data_one_mis_list
    
    def generate_nC2_data(self, query_list):
        data_double_nC2_list = np.array(list(map(double_nC2_seq_list,query_list)))
        return data_double_nC2_list
    
    def generate_distance_based_data(self,query_list):
        data_double_distance_based_list = np.array(list(map(generate_distance_based_seq_list,query_list)))
        return data_double_distance_based_list
    
    def data_concatenate(self,*args):
        data = []
        for _tmp_data in args:
            _tmp_data.reshape(-1,)
            data = np.append(data,_tmp_data)
        return data
    
    def __return_job_folder_path_N_make(self, job_name, path='.'):
        from pathlib import Path
        folder_path = f"{path}/{job_name}"
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        return folder_path
    
    def __seq_generate_log(self,log_path, log_name='Query_gen'):
        import logging
        from logging.handlers import RotatingFileHandler
        from pathlib import Path
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)
        #file_handler = logging.FileHandler(filename=log_dir,mode='a')
        #make folder contain log files
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=log_path,
            mode = 'a',
            maxBytes=10*1024*1024,
            backupCount=2,
        )
        formatter = logging.Formatter(
            fmt= "%(asctime)s - %(name)s - %(message)s"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        _ = list(map(logger.removeHandler,logger.handlers))
        assert not logger.hasHandlers()
        logger.addHandler(file_handler)
        return logger
    
    def __save_sequence(self, seq_data, file_name='query_seq.txt'):
        with open(file_name, 'w') as f:
            for _seq in seq_data:
                f.write(f"{_seq}\n")
    
    @classmethod
    def read_sequence_tolist(cls, file_name='query_seq.txt'):
        with open(file_name, 'r') as f:
            data = f.read().splitlines()
        return data


class generate_8_nC2_data(generate_mismatch_data):
    def __init__(self, num_set=1, threshold=7, **kwargs):
        self.num_set = num_set
        self.query_data = self.generate_query_seq_data(
            num_set = self.num_set,
            threshold= threshold,
            equal=False,
            **kwargs
        )
        self.mis_1_data = self.generate_one_mis_data(self.query_data)
        self.mis_2_data = self.generate_nC2_data(self.query_data)
        
