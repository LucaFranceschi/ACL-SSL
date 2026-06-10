# VGGSound-Sources dataset directory structure

```
[  14G]  VGGSS
├── [ 1.0K]  README.md
├── [  15K]  VGGSS_Dataset.py
├── [ 8.5G]  audio/ [5158 entries exceeds filelimit, not opening dir]
├── [ 7.6K]  eval_utils.py
├── [ 5.3G]  extend_audio/ [3471 entries exceeds filelimit, not opening dir]
├── [  14K]  extend_eval_utils.py
├── [ 207M]  extend_frames [5158 entries exceeds filelimit, not opening dir]
├── [ 216M]  frames [5158 entries exceeds filelimit, not opening dir]
├── [ 9.5M]  metadata
│   ├── [ 606K]  vggss.json
│   ├── [ 368K]  vggss_10k.csv
│   ├── [ 5.2M]  vggss_144k.csv
│   ├── [ 117K]  vggss_broad_classes.json
│   ├── [ 2.6M]  vggss_heard.csv
│   ├── [  96K]  vggss_heard_test.csv
│   ├── [ 192K]  vggss_test.csv
│   ├── [ 1.1K]  vggss_test_30.csv
│   ├── [ 3.7K]  vggss_test_100.csv
│   ├── [ 262K]  vggss_test_plus_silent.csv
│   └── [  95K]  vggss_unheard_test.csv
└── [ 4.2K]  unfold_dataset.ipynb
```

## Important
Official annotations (bounding box) are based on the 125th frame of a 25fps video for each file, not the exact center frame.
