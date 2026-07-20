import logging
import time
from datetime import datetime, timedelta
from transformers import Trainer, TrainingArguments, TrainerCallback
from model import load_base, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ProgressLogger(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.last_log_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        total_steps = state.max_steps
        logger.info(f"Starting training | Total steps: {total_steps} | Total epochs: {args.num_train_epochs}")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0 and state.global_step > self.last_log_step:
            elapsed = time.time() - self.start_time
            steps_done = state.global_step
            steps_per_sec = steps_done / elapsed
            remaining_steps = state.max_steps - steps_done
            eta_secs = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta_time = datetime.now() + timedelta(seconds=eta_secs)

            logger.info(f"Step {steps_done}/{state.max_steps} | Loss: {state.log_history[-1].get('loss', 'N/A'):.4f} | "
                       f"Elapsed: {elapsed/60:.1f}m | ETA: {eta_time.strftime('%H:%M:%S')} ({eta_secs/3600:.1f}h)")
            self.last_log_step = state.global_step

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        logger.info(f"Training complete | Total time: {elapsed/3600:.2f}h ({elapsed/60:.1f}m)")

model = load_base("gpt2-medium")
logger.info("Model loaded")

ds = load_dataset("data/items.csv")
logger.info(f"Dataset loaded | Size: {len(ds)}")

args = TrainingArguments(output_dir="ckpts", num_train_epochs=3,
                         per_device_train_batch_size=8,
                         logging_steps=10)
trainer = Trainer(model=model, args=args, train_dataset=ds, callbacks=[ProgressLogger()])

logger.info("Starting training...")
trainer.train()
