import logging
import time
from datetime import datetime, timedelta
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from model import load_base, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start = None
        self.train_start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.time()
        logger.info(f"Training started at {datetime.now().isoformat()}")
        logger.info(f"Total epochs: {int(args.num_train_epochs)}")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
        current_epoch = int(state.epoch) if state.epoch else 1
        logger.info(f"--- Epoch {current_epoch}/{int(args.num_train_epochs)} started ---")

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start
        current_epoch = int(state.epoch) if state.epoch else 1
        elapsed = time.time() - self.train_start
        remaining_epochs = int(args.num_train_epochs) - current_epoch
        estimated_remaining = epoch_time * remaining_epochs

        logger.info(f"Epoch {current_epoch} completed in {epoch_time:.1f}s")
        logger.info(f"Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)")
        if remaining_epochs > 0:
            eta_seconds = estimated_remaining
            eta_time = datetime.now() + timedelta(seconds=eta_seconds)
            logger.info(f"Estimated remaining time: {eta_seconds/60:.1f}m")
            logger.info(f"Estimated completion: {eta_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")

logger.info("Loading model and dataset...")
model = load_base("gpt2-medium")
ds = load_dataset("data/items.csv")
logger.info(f"Dataset size: {len(ds)} samples")

args = TrainingArguments(output_dir="ckpts", num_train_epochs=3,
                         per_device_train_batch_size=8,
                         logging_steps=10)
trainer = Trainer(model=model, args=args, train_dataset=ds,
                  callbacks=[ProgressCallback()])
logger.info("Starting training...")
trainer.train()
logger.info(f"Training completed at {datetime.now().isoformat()}")
