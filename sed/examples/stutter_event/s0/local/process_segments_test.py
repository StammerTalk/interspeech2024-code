# -*- coding: utf-8 -*-
import sys
import os

data_dir, category = sys.argv[1:]

os.makedirs(os.path.join(data_dir, category), exist_ok=True)

text = open(os.path.join(data_dir, category, 'text'), 'w')
segments = open(os.path.join(data_dir, category, 'segments'), 'w')

with open(os.path.join(data_dir, 'ref.flist') , 'r') as f:
    for ref_file in f:
        ref_file = ref_file.strip()
        if not ref_file: continue
        spk_id = os.path.splitext(os.path.basename(ref_file))[0]
        i, num_discard = 0, 0
        with open(ref_file) as fr:
            for line in fr:
                line = line.strip()
                if not line or line.startswith('Start,'): continue
                start, end, cate, p, b, s, wr, inter, *t = line.split(',')
                if cate != category:
                    num_discard += 1
                    continue
                utt_id = f'{spk_id}-{i:05d}'
                text.write(f'{utt_id} {",".join([p, b, s, wr, inter])}\n')
                segments.write(f'{utt_id} {spk_id} {start} {end}\n')
                i += 1
        print(spk_id, num_discard)

text.close()
segments.close()
