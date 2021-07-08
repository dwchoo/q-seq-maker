from seq_generator.query_generator import *
from seq_generator.mismatch_generator import *
import numpy as np

if __name__=="__main__":
    tmp_query = generate_query_seq_set_list()
    tmp_mis = double_nC2_seq_list('ACGTAAAAAAA')
    print(tmp_mis)
