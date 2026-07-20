import logging
import time
from datetime import datetime, timedelta
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from model import load_base, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ProgressLoggingCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("=" * 60)
        logger.info("Training started")
        logger.info(f"Total batches: {state.max_steps}")
        logger.info(f"Total epochs: {args.num_train_epochs}")
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % max(1, state.max_steps // 10) == 0:
            elapsed = time.time() - self.start_time
            progress = state.global_step / state.max_steps
            if progress > 0:
                eta_seconds = (elapsed / progress) - elapsed
                eta = datetime.now() + timedelta(seconds=eta_seconds)
                logger.info(
                    f"Step {state.global_step}/{state.max_steps} ({progress*100:.1f}%) | "
                    f"Loss: {state.log_history[-1].get('loss', 'N/A'):.4f} | "
                    f"ETA: {eta.strftime('%H:%M:%S')}"
                )

    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        eta_seconds = (elapsed / (state.epoch + 1)) * (args.num_train_epochs - state.epoch - 1)
        eta = datetime.now() + timedelta(seconds=eta_seconds)
        logger.info(f"Epoch {int(state.epoch)}/{args.num_train_epochs} complete | ETA: {eta.strftime('%H:%M:%S')}")

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        logger.info("=" * 60)
        logger.info(f"Training complete in {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {int(elapsed % 60)}s")


logger.info("Loading model and dataset...")
model = load_base("gpt2-medium")
ds = load_dataset("data/items.csv")
logger.info(f"Dataset size: {len(ds)} samples")

args = TrainingArguments(output_dir="ckpts", num_train_epochs=3,
                         per_device_train_batch_size=8)
trainer = Trainer(model=model, args=args, train_dataset=ds,
                 callbacks=[ProgressLoggingCallback()])
logger.info("Starting training...")
trainer.train()
