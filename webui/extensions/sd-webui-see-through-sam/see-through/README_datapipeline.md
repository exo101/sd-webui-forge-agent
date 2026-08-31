### Live2d extraction
Build the extraction program and extract examples following  
https://github.com/shitagaki-lab/CubismPartExtr

### Generate pseudo labels
Suppose you've placed the extraction directory at `workspace/datasets/partextr_output`, run

```
python inference/scripts/parse_live2d.py build_live2d_exec_list --srcd workspace/datasets/partextr_output

python inference/scripts/parse_live2d.py sam_infer_l2d --exec_list workspace/datasets/partextr_output/exec_list.txt

python inference/scripts/parse_live2d.py label_l2d_wsamsegs --exec_list workspace/datasets/partextr_output/exec_list.txt --extr_more
```

After that you can open `workspace/datasets/partextr_output/exec_list.txt` in the UI and correct the labels manually.

### Generate training data

Prepare the background images
```
cd workspace/datasets
hf download 24yearsold/anime_segmentation_bg --local-dir ./ --repo-type dataset
7z x anime_segmentation_bg.zip.001
rm -rf ./anime_segmentation_bg.zip.*

```

Run the synthesize script
```
python inference/scripts/syn_data.py render_body_samples --exec_list workspace/datasets/partextr_output/exec_list.txt --bg_list workspace/datasets/anime_segmentation_bg/exec_list.txt \
--save_dir workspace/datasets/test_bodysamples

python inference/scripts/syn_data.py get_tgt_list --src_dir workspace/datasets/partextr_bodysamples
```

