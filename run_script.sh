#/bin/bash
python analyser.py \
    -s ./data/other_paper_data/other_paper_Cpf1_digenome_seq.txt \
    --name other_digenome_seq \
    --PAM TTTV \
    --PAM_end 0 \
    --path ./data \
    --verbose
