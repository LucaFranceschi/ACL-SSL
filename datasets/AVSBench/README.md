# AVSBench dataset directory structure

```
datasets/AVSBench/
├── AVS1
│   ├── ms3
│   │   ├── audio_wav  [424 entries exceeds filelimit, not opening dir]
│   │   ├── gt_masks  [2120 entries exceeds filelimit, not opening dir]
│   │   └── visual_frames  [2120 entries exceeds filelimit, not opening dir]
│   ├── rename_files.py
│   ├── s4
│   │   ├── audio_wav  [4932 entries exceeds filelimit, not opening dir]
│   │   ├── gt_masks  [10852 entries exceeds filelimit, not opening dir]
│   │   └── visual_frames  [24660 entries exceeds filelimit, not opening dir]
│   └── unfold_dataset.sh
├── AVSBench_Dataset.py
├── README.md
├── eval_utils.py
├── metadata
│   ├── avs1_ms3_test.csv
│   └── avs1_s4_test.csv
├── ms3_data
│   ├── audio_log_mel
│   │   ├── test  [64 entries exceeds filelimit, not opening dir]
│   │   ├── train  [296 entries exceeds filelimit, not opening dir]
│   │   └── val  [64 entries exceeds filelimit, not opening dir]
│   ├── audio_wav
│   │   ├── test  [64 entries exceeds filelimit, not opening dir]
│   │   ├── train  [296 entries exceeds filelimit, not opening dir]
│   │   └── val  [64 entries exceeds filelimit, not opening dir]
│   ├── gt_masks
│   │   ├── test  [64 entries exceeds filelimit, not opening dir]
│   │   ├── train  [296 entries exceeds filelimit, not opening dir]
│   │   └── val  [64 entries exceeds filelimit, not opening dir]
│   ├── raw_videos  [424 entries exceeds filelimit, not opening dir]
│   └── visual_frames  [424 entries exceeds filelimit, not opening dir]
└── s4_data
    └── s4_data
        ├── audio_log_mel
        │   ├── test  [23 entries exceeds filelimit, not opening dir]
        │   ├── train  [23 entries exceeds filelimit, not opening dir]
        │   └── val  [23 entries exceeds filelimit, not opening dir]
        └── raw_videos
            ├── test  [23 entries exceeds filelimit, not opening dir]
            ├── train  [23 entries exceeds filelimit, not opening dir]
            └── val  [23 entries exceeds filelimit, not opening dir]

36 directories, 7 files
```

All .wav files sampled 16k

## Important
Fix bug in official test code (Issue: F-Score results vary depending on the batch number)

Considering the notable impact of this issue on the performance of self-supervised learning models, we suggest utilizing our updated test code.

We already discussed this issue with the author who released the official code.
