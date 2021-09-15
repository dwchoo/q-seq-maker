from seq_generator.query_generator import generate_ontarget_seq_set_list
from seq_generator.mismatch_generator import *
from seq_evaluator.evaluator import divide_seq_edit_distance
from encoder_decoder.seq_encoder_decoder import mismatch_calculator

import numpy as np
import pandas as pd
import pathlib
from pathlib import Path


class generate_mismatch_data:
    '''
    Generate mismatch data
    One mismatch
    nC2 mismatch
    distance based mismatch
    '''
    def generate_ontarget_seq_data(self,
        num_set,
        threshold,
        equal,
        log=True,
        job_name='ontarget_seq',
        save_path = '.'
    ):
        '''
        input:
            num_set    : number of ontarget set to generate
            threshold  : edit distance threshold 
            equal      : True, if last group has equal case
            log        : True, if save log
            job_name   : string, job name
            save_path  : save file path
        return:
            ontarget_set  : [ACGT,AACG ...]
        '''
        job_path = self.__return_job_folder_path_N_make(job_name=job_name, path=save_path)
        absolute_path = f"{Path().absolute()}"
        save_file_path = f"{job_path}/ontarget_{job_name}.txt"
        log_file_path = f"{job_path}/log/{job_name}.log"
        if log:
            logger = self.__seq_generate_log(log_file_path,'Ontarget')
            logger.info(f"Job information")
            logger.info(f"Job name: {job_name}")
            logger.info(f"path : {absolute_path}/{job_path}")
            logger.info(f"num_set : {num_set},  threshold : {threshold},  equal : {equal},  method : {type(self).__name__}")
        else:
            logger = None
            
        if equal:
            num_ontarget = num_set*12
        else:
            num_ontarget = num_set*8
            
        ontarget_set = []
        while 1:
            previous_ontarget_mass = np.array(ontarget_set).copy()
            new_ontarget_mass = generate_ontarget_seq_set_list(1, concate=True, equal=equal,join=True)
            pass_ontarget_mass, discard_ontarget_mass, _ = divide_seq_edit_distance(
                previous_ontarget_mass,
                new_ontarget_mass,
                threshold,
            )
            if len(discard_ontarget_mass) == 0:
                ontarget_set = pass_ontarget_mass.copy()
                if log:
                    logger.info(f"{len(pass_ontarget_mass)}/{num_ontarget} - save file : {save_file_path}")
                    self.__save_sequence(pass_ontarget_mass,save_file_path)
            else:
                ontarget_set = previous_ontarget_mass.copy()
            if len(ontarget_set) >= num_ontarget:
                pass_tmp, discard_tmp, _ = divide_seq_edit_distance(None, ontarget_set, threshold)
                if log:
                    logger.info(f"FINISH pass_num : {len(pass_tmp)}, discard_num : {len(discard_tmp)}")
                assert len(discard_tmp) == 0, f"discard seq num: {len(discard_tmp)}"
                return ontarget_set
    
    def generate_one_mis_data(self, ontarget_list):
        data_one_mis_list = np.array(list(map(one_mismatch_seq_list,ontarget_list)))
        return data_one_mis_list
    
    def generate_nC2_data(self, ontarget_list):
        data_double_nC2_list = np.array(list(map(double_nC2_seq_list,ontarget_list)))
        return data_double_nC2_list
    
    def generate_distance_based_data(self,ontarget_list):
        data_double_distance_based_list = np.array(list(map(generate_distance_based_seq_list,ontarget_list)))
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
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        #_ = list(map(logger.removeHandler,logger.handlers))
        #assert not logger.hasHandlers()
        logger.addHandler(file_handler)
        return logger
    
    def __save_sequence(self, seq_data, file_name='ontarget_seq.txt'):
        with open(file_name, 'w') as f:
            for _seq in seq_data:
                f.write(f"{_seq}\n")

    def __return_mismatch_data_csv(self, ontarget_list, offtarget_list_2d):
        df_data_list = []
        for _ontarget, _offtarget_list in zip(ontarget_list, offtarget_list_2d):
            for _offtarget in _offtarget_list:
                _mismatch_num = mismatch_calculator.seq_mismatch_count(_ontarget, _offtarget)
                _mismatch_info = mismatch_calculator.seq_mismatch_info_str(_ontarget, _offtarget)
                df_data_list.append([_ontarget,_offtarget,_mismatch_num, _mismatch_info])

        df_columns = ['on-target','off-target','number of mismatch','mismatch info']
        df_dtype = {
            'on-target' : 'category',' off-target' : 'category',
            'number of mismatch' : 'int8',}
        df_data = pd.DataFrame(
            df_data_list,
            columns = df_columns,
        ).astype(df_dtype)
        return df_data

    def __save_mismatch_data_csv(self,ontarget_list, offtarget_list_2d, file_path):
        df_data = self.__return_mismatch_data_csv(ontarget_list, offtarget_list_2d)
        df_data.to_csv(f'{file_path}.csv', index=False)

    
    @classmethod
    def read_sequence_tolist(cls, file_name='ontarget_seq.txt',check_file=True):
        with open(file_name, 'r') as f:
            data = f.read().splitlines()
        if check_file is False:
            return data
        else:
            new_data_list = []
            for _line in data:
                if '#' in _line:
                    continue
                else:
                    new_data_list.append(_line.upper())
            return new_data_list


class generate_8_nC2_data(generate_mismatch_data):
    def __init__(self, num_set=1, threshold=7, **kwargs):
        self.num_set = num_set
        self.ontarget_data = self.generate_ontarget_seq_data(
            num_set = self.num_set,
            threshold= threshold,
            equal=False,
            **kwargs
        )
        self.mis_1_data = self.generate_one_mis_data(self.ontarget_data)
        self.mis_2_data = self.generate_nC2_data(self.ontarget_data)
        self.mis_2_dis_data = self.generate_distance_based_data(self.ontarget_data)
 

class generate_12_data(generate_mismatch_data):
    def __init__(self, num_set=1, threshold=7, **kwargs):
        self.num_set = num_set
        self.ontarget_data = self.generate_ontarget_seq_data(
            num_set = self.num_set,
            threshold= threshold,
            equal=True,
            **kwargs
        )
        #self.mis_1_data = self.generate_one_mis_data(self.ontarget_data)
        #self.mis_2_data = self.generate_nC2_data(self.ontarget_data)
               
