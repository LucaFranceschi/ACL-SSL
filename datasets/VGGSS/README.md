# VGGSound-Sources dataset directory structure

```
datasets/VGGSS/
├── README.md
├── VGGSS_Dataset.py
├── audio  [5158 entries exceeds filelimit, not opening dir]
├── eval_utils.py
├── extend_audio  [3471 entries exceeds filelimit, not opening dir]
├── extend_eval_utils.py
├── extend_frames  [5158 entries exceeds filelimit, not opening dir]
├── frames  [5158 entries exceeds filelimit, not opening dir]
├── metadata
│   ├── vggss.json
│   ├── vggss_10k.csv
│   ├── vggss_144k.csv
│   ├── vggss_heard.csv
│   ├── vggss_heard_test.csv
│   ├── vggss_test.csv
│   ├── vggss_test_100.csv
│   ├── vggss_test_30.csv
│   ├── vggss_test_plus_silent.csv
│   └── vggss_unheard_test.csv
└── unfold_dataset.ipynb

6 directories, 15 files
```

All .wav files sampled 16k

## Important
Official annotations (bounding box) are based on the 125th frame of a 25fps video for each file, not the exact center frame.
