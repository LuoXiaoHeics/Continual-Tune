# Continual-Tune

In this work, we evaluate the LLMs during continual fine-tuning. 

For training, the codes are relatively simple. First, you need to  install the Python environment by ```pip install -r requirements.txt```.

Then a script example is shown in ```finetune.sh```, and you can just use bash to run the codes.

The processed data can be found in https://drive.google.com/drive/folders/1oqJ11w_3xGpBPXTmwJ1iz2LxtDHSrhxf?usp=sharing.

For evaluation, we adopt the evaluation framework of lm-evaluation-harness from https://github.com/EleutherAI/lm-evaluation-harness/tree/master. You can follow their instruction to build the test set. In our study, we test MMLU in 5-shots and other datasets in 0-shots. For example, you can run the bash scripts as follows:

```
python3 lm-evaluation-harness/main.py \
    --model hf-causal-experimental \
    --model_args pretrained=${path} \
    --tasks  piqa,boolq,winogrande,hellaswag,mathqa,mutual,toxigen \
    --device cuda:0 \
    --output_path results.txt \
    --no_cache 
```
