curr_dir=/
hf_model_dir=bigscience/bloomz-7b1

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
export TRANSFORMERS_CACHE=$curr_dir
export TORCH_EXTENSIONS_DIR=$curr_dir

deepspeed $curr_dir/scripts/train.py \
	--model_name_or_path $hf_model_dir \
	--data_path $curr_dir/instruction \
	--output_dir $curr_dir/bloom7b/ \
	--num_train_epochs 3 \
	--per_device_train_batch_size 4 \
	--gradient_accumulation_steps 4 \
	--save_steps 1000 \
	--save_strategy 'steps' \
	--save_total_limit 2 \
	--learning_rate 2e-5 \
    --warmup_steps 1 \
    --logging_steps 10 \
	--lr_scheduler_type 'constant' \
    --report_to 'tensorboard' \
    --gradient_checkpointing True \
    --deepspeed $curr_dir/configs/deepspeed_config_13B.json \
    --fp16 True 
	