import argparse, json, os

def get_args(logger):
    p = argparse.ArgumentParser(description="Backpack cross-lingual training")

    # Data
    p.add_argument("--model_id", type=str, default="")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--dataset_name", type=str, default="allenai/wmt22_african")
    p.add_argument("--dataset_config", type=str, default="afr-eng")     # e.g., "afr-eng"
    p.add_argument("--dataset_split", type=str, default="train")   # HF slice or split name
    p.add_argument("--validation_dataset", type=str, default="facebook/flores")

    # Loss weights/temps
    p.add_argument("--normalize_last_token_embeds", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--normalize_sense_pooling", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--add_lm_loss", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--align_pct", type=float, default=0.20,
                   help="Fraction of total steps for early contrastive-dominant phase.")
    p.add_argument("--polish_pct", type=float, default=0.15,
                   help="Fraction of total steps for LM-dominant final phase.")
    p.add_argument("--sense_pool_temp", type=float, default=1.0, help="Temperature for sense pooling")

    # weights at phase boundaries
    p.add_argument("--freeze_sense_during_polish", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--w_ctx_align", type=float, default=0.45,
                   help="Context loss weight during align phase.")
    p.add_argument("--w_sns_align", type=float, default=0.55,
                   help="Sense loss weight during align phase.")
    p.add_argument("--w_lm_align",  type=float, default=0.02,
                   help="LM loss weight during align phase.")

    p.add_argument("--w_ctx_mid", type=float, default=0.40,
                   help="Context loss weight during middle (joint) phase.")
    p.add_argument("--w_sns_mid", type=float, default=0.40,
                   help="Sense loss weight during middle (joint) phase.")
    p.add_argument("--w_lm_mid",  type=float, default=0.20,
                   help="LM loss weight during middle (joint) phase.")

    p.add_argument("--w_ctx_tail", type=float, default=0.15,
                   help="Context loss weight during LM-dominant tail.")
    p.add_argument("--w_sns_tail", type=float, default=0.15,
                   help="Sense loss weight during LM-dominant tail.")
    p.add_argument("--w_lm_tail",  type=float, default=0.70,
                   help="LM loss weight during LM-dominant tail.")

    # InfoNCE temperature annealing
    p.add_argument("--tau_ctx_start", type=float, default=0.07,
                   help="Starting temperature for context InfoNCE.")
    p.add_argument("--tau_ctx_end",   type=float, default=0.04,
                   help="Ending temperature for context InfoNCE.")
    p.add_argument("--tau_sns_start", type=float, default=0.05,
                   help="Starting temperature for sense InfoNCE.")
    p.add_argument("--tau_sns_end",   type=float, default=0.03,
                   help="Ending temperature for sense InfoNCE.")

    # Optimization
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--test_batch_size", type=int, default=32)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)

    # Runtime
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")
    p.add_argument("--use_fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_steps", type=int, default=40_000)
    p.add_argument("--save_dir", type=str, default="")
    p.add_argument("--overwrite_output_dir",
        action="store_true",
        help="If passed, overwrite the contents of the output directory (save_dir).",
    )

    # Tracking
    p.add_argument("--log_every_n_steps", type=int, default=0)
    p.add_argument("--eval_every_n_steps", type=int, default=2000)
    p.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=False,
               help="Enable Weights & Biases logging.")
    p.add_argument("--wandb_project", type=str, default="backpack-xalign",
                help="Weights & Biases project name.")
    p.add_argument("--wandb_run_name", type=str, default=None,
                help="Optional run name for Weights & Biases.")
    p.add_argument(
        "--wandb_run_id",
        type=str,
        default="",
        help="Stable Weights & Biases run id for resuming the same experiment.",
    )
    
    # Checkpointing
    p.add_argument("--save_every", type=int, default=0,
               help="Save a checkpoint every N optimizer steps (0=off).")
    p.add_argument("--max_checkpoints", type=int, default=3,
                help="Max number of checkpoints to keep (0=keep all).")
    p.add_argument("--resume_from_checkpoint", action=argparse.BooleanOptionalAction, default=False,
                help="Resume from the most recent checkpoint in save_dir.")

    # Optional: override source/target 
    p.add_argument("--src", type=str, default=None, help="Override source lang (e.g., 'afr')")
    p.add_argument("--tgt", type=str, default=None, help="Override target lang (e.g., 'eng')")

    args = p.parse_args()

    # Derive src/tgt from dataset_config if not provided
    if args.src is None or args.tgt is None:
        try:
            src, tgt = args.dataset_config.split("-")
        except ValueError:
            raise ValueError(f"dataset_config must look like 'src-tgt', got: {args.dataset_config}")
        if args.src is None: args.src = src
        if args.tgt is None: args.tgt = tgt

    # Make save_dir and save the config for reproducibility
    if os.path.exists(args.save_dir):
        if args.overwrite_output_dir:
            logger.info(f"--overwrite_output_dir requested for {args.save_dir}.")

        elif getattr(args, "resume_from_checkpoint", False):
            # Resume mode: don't delete or raise, just reuse directory
            logger.info(f"Resuming from existing directory: {args.save_dir}")
            if not os.path.exists(os.path.join(args.save_dir, "checkpoints")):
                logger.warning(
                    f"--resume_from_checkpoint was set but no 'checkpoints/' folder found in {args.save_dir}. "
                    "Starting from scratch."
                )

        else:
            # directory exists and no overwrite/resume => safety check
            if os.listdir(args.save_dir):
                raise ValueError(
                    f"Output directory {args.save_dir} already exists and is not empty. "
                    "Use --overwrite_output_dir to clear it or --resume_from_checkpoint to resume."
                )
    else:
        os.makedirs(args.save_dir, exist_ok=True)


    with open(os.path.join(args.save_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    
    return args