# Continual-Tune

In this work, we evaluate the LLMs during continual fine-tuning. 

For training, the codes are relatively simple. First, you need to install the Python environment by ```pip install -r requirements.txt```.

Then a script example is shown in ```finetune.sh```, and you can directly use bash to run the codes.

The processed data can be found at https://drive.google.com/drive/folders/1oqJ11w_3xGpBPXTmwJ1iz2LxtDHSrhxf?usp=sharing. We mainly adopt the instruction tasks used in Scialom et al [1], which can also be found at https://github.com/ThomasScialom/T0_continual_learning.

For evaluation, we adopt the evaluation framework of lm-evaluation-harness from https://github.com/EleutherAI/lm-evaluation-harness/tree/master. You can follow their instruction to build the test environment. Our study tests MMLU in 5-shots and other datasets in 0-shots. For example, you can run the bash scripts as follows:

```
python3 lm-evaluation-harness/main.py \
    --model hf-causal-experimental \
    --model_args pretrained=${path} \
    --tasks  piqa,boolq,winogrande,hellaswag,mathqa,mutual \
    --device cuda:0 \
    --output_path results.txt \
    --no_cache 
```

Note that in lm-evaluation-harness, some tasks like MMLU are evaluated separately for each split. Thus some codes are required to merge the splits. We use ```datasets.concatenate_datasets``` and create new classes following their instruction to implement this step. 

[1] Thomas Scialom, Tuhin Chakrabarty, and Smaranda Muresan. 2022. Fine-tuned Language Models are Continual Learners. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 6107–6122, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics


## Citation

@article{luo2023empirical,

title={An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning}, 
      
author={Yun Luo and Zhen Yang and Fandong Meng and Yafu Li and Jie Zhou and Yue Zhang},
      
year={2023},      
      
eprint={2308.08747},    
      
archivePrefix={arXiv}

}
