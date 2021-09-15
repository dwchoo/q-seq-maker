#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

class crispr:
    def __init__(self, length, PAM, PAM_end=True):
        self.N_seq = 'N'*length
        self.PAM = PAM
        self.PAM_end = PAM_end
        PAM_length = len(PAM)
        self.N_PAM = 'N'*PAM_length

    def search_space(self,):
        if not self.PAM_end:
            return f"{self.PAM}{self.N_seq}"
        else:
            return f"{self.N_seq}{self.PAM}"

    def search_seq(self, seq):
        if not self.PAM_end:
            return f"{self.N_PAM}{seq}"
        else:
            return f"{seq}{self.N_PAM}"

        
#class cas9(crispr):
#    def __init__(self, )
