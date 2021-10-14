#/bin/bash
#TYPE=111
for TYPE in 112 121 122 211 212 221 222
do
    python analyser.py \
        -s ./data/other_paper_offtarget/type_$TYPE/seq_list_type_$TYPE.txt \
        --name type_$TYPE \
        --PAM TTTV \
        --PAM_end 0 \
        --path ./data/other_paper_offtarget \
        --verbose
done
