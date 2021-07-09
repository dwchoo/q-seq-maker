from seq_generator.query_generator import *
from seq_generator.mismatch_generator import *
import numpy as np

if __name__=="__main__":
    tmp_data = generate_8_nC2_data(
        num_set=1,
        threshold=7,
        log=True,
        job_name='test',
        save_path='.'
    )
    print(tmp_data.query_data)
