#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import glob
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer,EvalPrediction
import json
import copy

import numpy as np
import os


from data_handler import continual_train


os.environ['MASTER_PORT'] = '12340'


task = ['wiki_auto','empathetic_dialogues','eli5','eSNLI','gigaword']

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"



instruction_datasets = ['ag_news','anli','common_gen','glue_mrpc','glue_qqp','imdb','rotten_tomatoes','rte','samsum','trec',
                        'winogrande','wsc','xsum']




@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def minus_data(dataset, train_num = 100000, seed=0):
    np.random.seed(seed)

    size = len(dataset['input_ids'])
    idxs = np.array(range(len(dataset['input_ids'])))
    np.random.shuffle(idxs)

    train_idx = idxs[:train_num]

    return train_idx

def train_val_split(dataset, train_percent, seed=0):
    np.random.seed(seed)

    size = len(dataset['input_ids'])
    idxs = np.array(range(len(dataset['input_ids'])))
    np.random.shuffle(idxs)

    train_idx = idxs[:int(size*train_percent)]
    test_idx = idxs[int(size*train_percent):]

    return train_idx, test_idx


def get_data_by_idx(dataset, idxs):
    input_id = []
    labels = []
    for item_id in idxs:
        labels.append(dataset['labels'][item_id])
        input_id.append(dataset['input_ids'][item_id])
    return input_id, labels


class InstructionDataset_S(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, split = 'train',task = 'ag_news'):
        super(InstructionDataset_S, self).__init__()
        logging.warning("Loading data...")

        self.input_ids = []
        self.labels = []

        task_splits = continual_train[task]['train']
        for u in task_splits:
            with open(data_path+'/'+u[1]+'.train.json', 'r') as f:
                data = json.load(f)
                
            logging.warning("Formatting inputs...")
            
            sources = [f"Below is an instruction that describes a task, paired with an input that provides  \
                       further context. Write a response that appropriately completes the request.\n\n{example}\n\n### Response:" for example in data['src']]
            deals = [str(u).replace("['","").replace("']","") for u in data['tgt']]
            targets = [f"{example}{tokenizer.eos_token}" for example in deals]

            logging.warning("Tokenizing inputs... This may take some time...")
            data_dict = preprocess(sources, targets, tokenizer)

            train_idx = minus_data(data_dict,int(100000/len(task_splits)))
            input_ids,labels = get_data_by_idx(data_dict,train_idx)
            print(len(input_ids))
            self.input_ids.extend(input_ids)
            self.labels.extend(labels)

        
        print('The number of data samples: ' + str(len(self.input_ids)))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def load_latest_ckpt(output_path):
    all_ckpts = glob.glob(
            f"{output_path}/checkpoint-*"
        )
    print(all_ckpts)
    steps = [int(u.split('/')[-1].split('-')[1]) for u in all_ckpts]
    ckpt_list = sorted(
        steps, reverse=True
    )
    assert len(ckpt_list) > 0, f"no checkpoint found"
    return 'checkpoint-'+str(ckpt_list[0])

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path,t) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = InstructionDataset_S(tokenizer=tokenizer, data_path=data_path,task=t)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    


    for i in range(len(task)):
        if os.path.exists(training_args.output_dir+task[i]):
            continue
        t_args = copy.deepcopy(training_args)
        t = task[i]
        t_args.output_dir += t+'/'
        if i == 0:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
            )

            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=special_tokens_dict,
                tokenizer=tokenizer,
                model=model,
            )

            data_module = make_supervised_data_module(tokenizer=tokenizer, data_path=data_args.data_path+'/'+t,t=t)
            trainer = Trainer(model=model, tokenizer=tokenizer, args=t_args, **data_module)
            trainer.train()
            trainer.save_state()
            safe_save_model_for_hf_trainer(trainer=trainer, output_dir=t_args.output_dir)
            del data_module
            del trainer
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            
        else:
            model_name = load_latest_ckpt(training_args.output_dir+task[i-1])
            model = transformers.AutoModelForCausalLM.from_pretrained(
                training_args.output_dir+task[i-1]+'/'+model_name,
            )

            # smart_tokenizer_and_embedding_resize(
            #     special_tokens_dict=special_tokens_dict,
            #     tokenizer=tokenizer,
            #     model=model,
            # )

            data_module = make_supervised_data_module(tokenizer=tokenizer, data_path=data_args.data_path+'instruction/'+t,t=t)
            # del trainer
            trainer = Trainer(model=model, tokenizer=tokenizer, args=t_args, **data_module)
            trainer.train()
            trainer.save_state()
            safe_save_model_for_hf_trainer(trainer=trainer, output_dir=t_args.output_dir)


if __name__ == "__main__":
    train()
