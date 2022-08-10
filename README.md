# QATest

QATest is a tool using novel sentence-level mutation based metamorphic testing, which can precisely detect bugs in
QA software.

## Reproducibility

### Environment build

1. download all datasets from this [Link](https://drive.google.com/drive/folders/18tbGI89R3S9YIYRPHxIZcv6drQCJZ6RE?usp=sharing) and put it in the root directory
2. build SLAHAN environment following the tutorial [SLAHAN](https://github.com/kamigaito/SLAHAN)
3. install all python dependent packages for QATest
```
pip install -r requirements.txt
```
4. unzip the file benepar_en3.zip in 3rd_models
```
cd 3rd_models
unzip benepar_en3.zip
```


### Run QATest

1. run SLAHAN to compress the seed question into a short question. 
The compressed questions are saved in path `datasets/compress`

2. run QATest to generate all new test cases and detected bugs. 
The results are saved in path `QATest/results/`

```
cd script/
python run.py project_name  # valid project_name are 'boolq', 'squad2', 'narrative' 
```







----

## Manual Labeling Results

For the results of manual labeling about **false positive** and **naturalness** , it were placed in the directory [labeling_results](./labeling_results)

