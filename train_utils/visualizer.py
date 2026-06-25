import matplotlib.pyplot as plt
from pathlib import Path
import json
import os


class TrainingVisualizer:
    def __init__(self, save_dir="./training_plots"):
        self.save_dir = Path(save_dir)
        self.metrics = {
            'train_loss': [],
            'val_dice': [],
            'lr': []
        }
        self._setup_dirs()

    def _setup_dirs(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def update_metrics(self, epoch, train_loss, val_dice, lr):

        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_dice'].append(val_dice)
        self.metrics['lr'].append(lr)
        

        self._save_metrics()

    def _save_metrics(self):

        metrics_file = self.save_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f)

    def load_metrics(self, resume_epoch=None):

        metrics_file = self.save_dir / "training_metrics.json"
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                saved_metrics = json.load(f)
                
                if resume_epoch is not None:

                    max_idx = min(resume_epoch, len(saved_metrics['train_loss']))
                    self.metrics['train_loss'] = saved_metrics['train_loss'][:max_idx]
                    self.metrics['val_dice'] = saved_metrics['val_dice'][:max_idx]
                    self.metrics['lr'] = saved_metrics['lr'][:max_idx]
                else:
                    self.metrics = saved_metrics
                
                print(f"The previous training metrics have been loaded, covering a total of {len(self.metrics['train_loss'])} rounds")
                return True
        return False

    def plot_final_metrics(self):

        plt.figure(figsize=(15, 5))


        plt.subplot(1, 3, 1)
        plt.plot(self.metrics['train_loss'], 'b-o', linewidth=2, markersize=1)
        plt.title("Training Loss", fontsize=12)
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Loss", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)


        plt.subplot(1, 3, 2)
        plt.plot(self.metrics['val_dice'], 'r-s', linewidth=2, markersize=1)
        plt.title("Validation Dice", fontsize=12)
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Dice Coefficient", fontsize=10)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)


        plt.subplot(1, 3, 3)
        plt.plot(self.metrics['lr'], 'g-^', linewidth=2, markersize=1)
        plt.title("Learning Rate Schedule", fontsize=12)
        plt.xlabel("Epoch", fontsize=10)
        

        lr_max = max(self.metrics['lr']) if self.metrics['lr'] else 0
        lr_min = min(self.metrics['lr']) if self.metrics['lr'] else 0
        
        if lr_max > 0 and lr_min > 0 and lr_max / lr_min > 10:
            plt.ylabel("LR (log scale)", fontsize=10)
            plt.yscale('log')
        else:
            plt.ylabel("Learning Rate", fontsize=10)
            
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)


        plt.tight_layout(pad=3.0)
        plt.savefig(self.save_dir / "MoNuSeg_Unet-4_final-metrics.png",
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white')
        plt.close()