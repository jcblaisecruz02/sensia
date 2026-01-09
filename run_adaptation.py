from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, get_scheduler
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler

from info_nce import InfoNCE

from utils.data import preprocess_function, collate_fn, load_flores_pair, preprocess_flores, FLORES_CODE_MAP, make_dataloader
from utils.training import train_model, LossScheduler, LossScheduleCfg
from utils.helpers import log_param_breakdown, log_trainable_params, set_seed, maybe_resume
from utils.args import get_args

from utils.helpers import init_distributed, cleanup_distributed, is_main_process, get_world_size, barrier
from torch.nn.parallel import DistributedDataParallel as DDP

import os, logging, uuid, shutil
import wandb


if __name__ == "__main__":
    # setup logger
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)

    # get args
    args = get_args(logger)
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #set_seed(args.seed)
    init_distributed(logger)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_seed(getattr(args, "seed", 42))

    # Handle deletion
    if getattr(args, "overwrite_output_dir", False):
        if is_main_process():
            if os.path.isdir(args.save_dir):
                logger.warning(f"--overwrite_output_dir active, deleting {args.save_dir}")
                # ignore_errors=True avoids races with lingering temp files
                shutil.rmtree(args.save_dir, ignore_errors=True)
            os.makedirs(args.save_dir, exist_ok=True)
        barrier()
    else:
        # Non-overwrite paths: ensure directory exists (rank-0), then sync
        if is_main_process():
            os.makedirs(args.save_dir, exist_ok=True)
        barrier()

    # config flores codes
    flores_src = FLORES_CODE_MAP[args.src]
    flores_tgt = FLORES_CODE_MAP[args.tgt]

    # Load the model
    logger.info("Loading the model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    torch_model = AutoModelForCausalLM.from_pretrained(args.model_id, config=config, trust_remote_code=True)

    # move to device
    torch_model = torch_model.to(device)
    torch_model.train()

    # attempt to compile the model if possible
    try:
        model = torch.compile(torch_model, mode="reduce-overhead")  # or "max-autotune"
    except Exception:
        if logger: logger.info("torch.compile not applied (unsupported graph).")

    # DDP Wrap
    if get_world_size() > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True) 

    # Add the pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_model.config.pad_token_id = tokenizer.pad_token_id

    # load the dataset
    logger.info("Loading dataset and preprocessing")
    train_dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    flores_ds = load_flores_pair(flores_src, flores_tgt, split="devtest", dataset=args.validation_dataset)

    # Preprocess the dataset
    tokenized_train_dataset = train_dataset.map(
        preprocess_function,
        batched=False,
        fn_kwargs={
            "tokenizer": tokenizer,
            "src": args.src,
            "tgt": args.tgt,
            "max_length": args.max_length,
        },
        remove_columns=train_dataset.column_names  # removes 'translation', etc.
    )
    tokenized_flores = flores_ds.map(
        preprocess_flores,
        batched=False,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": args.max_length,
        },
        remove_columns=flores_ds.column_names,  # keep only tokenized fields
    )

    # Make the dataloader
    train_dataloader = make_dataloader(
        tokenized_train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last_train=True
    )
    if get_world_size() > 1:
        if is_main_process():
            test_dataloader = DataLoader(
                tokenized_flores,
                batch_size=args.test_batch_size,
                shuffle=False,
                collate_fn=lambda batch: collate_fn(batch, tokenizer)
            )
        else: test_dataloader = None
    else:
        test_dataloader = DataLoader(
            tokenized_flores,
            batch_size=args.test_batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, tokenizer)
        )

    # optimizer and loss
    logger.info("Setting up training")
    optimizer = AdamW(torch_model.parameters(), lr=args.learning_rate)
    warmup_steps = int(args.max_steps * args.warmup_ratio)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_steps,
    )

    sense_loss_fn = InfoNCE(temperature=args.tau_sns_start)
    context_loss_fn = InfoNCE(temperature=args.tau_ctx_start)
    scaler = GradScaler()

    # Configure the scheduler for loss weights
    cfg = LossScheduleCfg(
        align_pct=args.align_pct,
        polish_pct=args.polish_pct,
        w_ctx_align=args.w_ctx_align,
        w_sns_align=args.w_sns_align,
        w_lm_align=args.w_lm_align,
        w_ctx_mid=args.w_ctx_mid,
        w_sns_mid=args.w_sns_mid,
        w_lm_mid=args.w_lm_mid,
        w_ctx_tail=args.w_ctx_tail,
        w_sns_tail=args.w_sns_tail,
        w_lm_tail=args.w_lm_tail,
        tau_ctx_start=args.tau_ctx_start,
        tau_ctx_end=args.tau_ctx_end,
        tau_sns_start=args.tau_sns_start,
        tau_sns_end=args.tau_sns_end,
    )
    scheduler = LossScheduler(cfg)

    # Resuming logic
    if get_world_size() > 1: barrier()
    start_step = 0
    start_step, extra_state = maybe_resume(
        args,
        model=torch_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        context_loss_fn=context_loss_fn,
        sense_loss_fn=sense_loss_fn,
        loss_scheduler=scheduler,
        logger=logger,
    )
    if get_world_size() > 1: barrier()

    # wandb
    # if user passed a fixed id, use it; else generate one
    # Defaults on every rank
    ckpt_run_id = None                 # will stay None unless you load it from a checkpoint
    cli_run_id  = os.environ.get("WANDB_RUN_ID")  # optional, may be None
    wandb_run   = None
    
    if args.use_wandb and is_main_process():
        # Prefer run id from checkpoint; else from CLI; else generate new.
        ckpt_run_id = (extra_state or {}).get("wandb_run_id")
        cli_run_id  = getattr(args, "wandb_run_id", None) or None
        run_id = ckpt_run_id or cli_run_id or uuid.uuid4().hex

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=run_id,
            resume="allow" if (ckpt_run_id or cli_run_id) else None,  # resume if we had an id already
            config=vars(args),
        )

    # Keep this around so saves include the id (lets future resumes recover it)
    wandb_run_id = (wandb_run.id if wandb_run else ckpt_run_id) or cli_run_id
    wandb_state  = {"wandb_run_id": wandb_run_id}   

    # training loop
    logger.info("Beginning training")
    logger.info(f"Total number of training steps: {args.max_steps:,}")

    log_trainable_params(torch_model, logger)
    log_param_breakdown(torch_model.backpack, logger)

    train_model(
        model=torch_model,
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        tokenizer=tokenizer,
        device=device,
        total_steps=args.max_steps,
        context_loss_fn=context_loss_fn,
        sense_loss_fn=sense_loss_fn,
        scheduler=scheduler,
        normalize_last_token_embeds=args.normalize_last_token_embeds,
        normalize_sense_pooling=args.normalize_sense_pooling,
        freeze_sense_during_polish=args.freeze_sense_during_polish,
        add_lm_loss=args.add_lm_loss,
        sense_pool_temp=args.sense_pool_temp,
        clip_grad_norm=args.clip_grad_norm,
        label_smoothing=args.label_smoothing,
        use_fp16=args.use_fp16,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        eval_every_n_steps=args.eval_every_n_steps,
        use_wandb=args.use_wandb,
        start_step=start_step,
        args=args,
    )

    if is_main_process():
        logger.info("Done training, saving model")
        os.makedirs(args.save_dir, exist_ok=True)
        torch_model.save_pretrained(args.save_dir, safe_serialization=False)
        tokenizer.save_pretrained(args.save_dir, safe_serialization=False)

    if wandb_run is not None and is_main_process():
        wandb.finish()

    cleanup_distributed()