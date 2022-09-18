# QAQA: Natural Test Generation for Precise Testing of Question Answering Software
QAQA is a QA software testing technique. It can generate natural test inputs and achieve precise testing.

QAQA is the artifact for paper [Natural Test Generation for Precise Testing of Question Answering Software](https://github.com/ShenQingchao/QAQA/blob/master/Natural%20Test%20Generation%20for%20Precise%20Testing%20of%20Question%20Answering%20Software.pdf), which have accepted by ASE'22.



## Reproducibility

### Environment build

1. download all datasets from this [Link](https://drive.google.com/drive/folders/18tbGI89R3S9YIYRPHxIZcv6drQCJZ6RE?usp=sharing) and put it in the root directory
2. build SLAHAN environment following the tutorial [SLAHAN](https://github.com/kamigaito/SLAHAN)
3. install all python dependent packages for QAQA
```
pip install -r requirements.txt
```
4. unzip the file benepar_en3.zip in 3rd_models
```
cd 3rd_models
unzip benepar_en3.zip
```


### Run QAQA

1. run SLAHAN to compress the seed question into a short question. 
The compressed questions are saved in path `datasets/compress`

2. run QAQA to generate all new test cases and detected bugs. 
The results are saved in path `QAQA/results/`

```
cd script/
python run.py project_name  # valid project_name are 'boolq', 'squad2', 'narrative' 
```


----

## Manual Labeling Results

For the results of manual labeling about **false positive** and **naturalness** , it were placed in the directory [labeling_results](./labeling_results)

-----

## Citation
Please cite our paper if this work is useful for you.

