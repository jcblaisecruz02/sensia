# Multilinguality as Sense Adaptation

This repository contains code for our paper **Multilinguality as Sense Adaptation**.

## Quick Navigation
- [Data Preparation (OPUS + HF Mix)](#data-preparation-opus--hf-mix)
- [Backpack Pretraining / Retuning](#backpack-pretraining--retuning)
- [Sense Adaptation](#sense-adaptation)
- [Benchmarking](#benchmarking)

## Data Preparation (OPUS + HF Mix)
Build mixed OPUS/HF corpora with `build_opus_corpus.py`. Point `--hf_out` and `--work_dir` to your storage, select corpora, and tune the filtering thresholds.

```bash
python build_opus_corpus.py \
	--src eng --tgt swh \
	--hf_out $SCRATCH/data-output/en-swh-hf \
	--work_dir $SCRATCH/data-output/work-dir/en-swh \
	--corpora CCMatrix CCAligned JW300 GlobalVoices QED OpenSubtitles TED2013 TED2020 WikiMatrix WikiTitles \
	--hf_corpora allenai/nllb allenai/wmt22_african \
	--hf_corpora_split train \
	--max_per_corpus 500000 \
	--batch_size 2048 \
	--min_len 3 --max_len 256 \
	--len_ratio_low 0.45 --len_ratio_high 2.0 \
	--labse_threshold 0.74 \
	--target_size 100000 \
	--opus_download_dir $SCRATCH/data-output/opus_cache \
	--log_level INFO \
	--upload_data \
	--config_name eng-swh-100k
```

## Backpack Pretraining / Retuning
We have a modified `run_clm.py` that you can use to retrain a backpack model if needed.

```bash
export WANDB_PROJECT=<your_project_name>
python run_clm.py \
	--model_name_or_path <your_model_config> \ 
	--config_name <your_model_config> \
	--tokenizer_name <your_model_config> \
	--dataset_name Skylion007/openwebtext \
	--per_device_train_batch_size 64 \
	--per_device_eval_batch_size 64 \
	--bf16 \
	--gradient_accumulation_steps 32 \
	--warmup_ratio 0.1 \
	--weight_decay 0.01 \
	--do_train --do_eval \
	--output_dir <output> \
	--max_steps 300000 \
	--block_size 512 \
	--report_to wandb \
	--run_name <runname> \
	--logging_steps 32 \
	--save_steps 512 \
	--save_total_limit 5
```

## Sense Adaptation
Fine-tune Backpack models with sense-adaptation schedules via `run_adaptation.py`. Adjust the dataset config, loss schedule, run length, and checkpoint paths to match your experiment.

```bash
export WANDB_PROJECT=<your_project_name>
python run_adaptation.py \
	--model_id <backpack_model> \
	--max_length 256 \
	--dataset_name <dataset_from_last_last_step> \
	--dataset_config <your_config> \
	--dataset_split train \
	--src eng --tgt swh \
	--normalize_sense_pooling \
	--normalize_last_token_embeds \
	--add_lm_loss \
	--learning_rate 5e-5 \
	--train_batch_size 64 \
	--test_batch_size 64 \
	--warmup_ratio 0.02 \
	--clip_grad_norm 1.0 \
	--label_smoothing 0.05 \
	--sense_pool_temp 0.7 \
	--align_pct 0.20 --polish_pct 0.50 \
	--w_ctx_align 0.45 --w_sns_align 0.55 --w_lm_align 0.02 \
	--w_ctx_mid 0.40 --w_sns_mid 0.40 --w_lm_mid 0.20 \
	--w_ctx_tail 0.15 --w_sns_tail 0.15 --w_lm_tail 0.70 \
	--tau_ctx_start 0.07 --tau_ctx_end 0.07 \
	--tau_sns_start 0.05 --tau_sns_end 0.05 \
	--use_fp16 \
	--max_steps 150000 \
	--eval_every_n_steps 2000 \
	--save_every 2000 \
	--max_checkpoints 1 \
	--log_every_n_steps 100 \
	--resume_from_checkpoint \
	--save_dir <your_save_dir> \
	--use_wandb \
	--wandb_run_name <run_name> \
	--seed 1234
```
Models are also available [in HuggingFace](https://hf.co/collections/jcblaise/multilinguality-as-sense-adaptation) if you don't want to make your own.

## Benchmarking
Evaluate FLORES-200 perplexity with `run_flores_ppl.py`. Choose the FLORES language tag, adjust `--max_length`, and point `--model_path` at a local or hub checkpoint.

```bash
python run_flores_ppl.py \
	--model_path <your_model> \
	--lang swh_Latn \
	--split devtest \
	--batch_size 64 \
	--max_length 256 \
	--device cuda \
	--fp16
```
`run_eval.py` covers all tasks via `--task`. Pick the scoring mode (conditional likelihood, PMI, option letter) and the language-specific hyperparameters.

```bash
python run_eval.py \
	--model_path <your_model> \
	--task xcopa \
	--lang swh \
	--split test \
	--batch_size 64 \
	--fp16 
```
