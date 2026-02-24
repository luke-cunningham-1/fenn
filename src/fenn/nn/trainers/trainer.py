from typing import Optional, Union, List
import torch
from pathlib import Path
#from pathlib import Path

from fenn.logging import Logger

class Trainer:
    """The base Trainer class"""

    def __init__(self,
                 model,
                 loss_fn,
                 optim,
                 epochs,
                 device="cpu",
                 checkpoint_dir: Optional[Union[Path, str]] = None, 
                 checkpoint_epochs: Optional[Union[int, List[int]]] = None,
                 checkpoint_name: str = "checkpoint",
                 save_best: bool = False
                 ):
        

        self._logger = Logger()

        self._device = device

        self._model = model.to(device)
        self._model.train()
        self._loss_fn = loss_fn
        self._optimizer = optim
        self._epochs = epochs
        self._metrics = {}

        # chechpoint setup 
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._checkpoint_epochs = checkpoint_epochs
        self._checkpoint_name = checkpoint_name
        self._save_best = save_best
        self._best_loss = float('inf')

        # create the checkpoint directory if it doesn't exist and is enabled
        if self._checkpoint_dir and (self._checkpoint_epochs is not None or self._save_best):
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if self._checkpoint_epochs is not None:
                self._logger.system_info(f"Checkpointing enabled. Checkpoints will be saved to {self._checkpoint_dir} every {self._checkpoint_epochs} epochs.")
            if self._save_best:
                self._logger.system_info(f"Best model checkpointing enabled. Best model will be saved to {self._checkpoint_dir}.")


    def _should_save_checkpoint(self, epoch: int):
        """Determine if a checkpoint should be saved at the given epoch.
        
        Args: 
            epoch (int): The current epoch number. (0-indexed)
        Returns:
            bool: True if a checkpoint should be saved, False otherwise.
        
        """
        if self._checkpoint_dir is None or self._checkpoint_epochs is None:
            return False

        if isinstance(self._checkpoint_epochs, int):
            # save every N epochs 
            return epoch % self._checkpoint_epochs == 0 or epoch == self._epochs-1
        elif isinstance(self._checkpoint_epochs, list):
            # save at specific epochs
            return epoch in self._checkpoint_epochs
        return False
    
    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False): 
        """Save a checkpoint of the model at the given epoch.
        
        Args:
            epoch (int): The current epoch number. (0-indexed)
            loss: training loss for this epoch 
            is_best: if true save as best model
        """
        if self._checkpoint_dir is None:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'loss': loss,
            'best_loss': self._best_loss
        }

        if not is_best:
            filename = f"{self._checkpoint_name}_epoch_{epoch}.pt"
            filepath = self._checkpoint_dir / filename
            torch.save(checkpoint, filepath)
            self._logger.system_info(f"Checkpoint saved at epoch {epoch} to {filepath}.")
        
        if is_best and self._save_best:
            best_filepath = self._checkpoint_dir / f"{self._checkpoint_name}_best.pt"
            torch.save(checkpoint, best_filepath)
            self._logger.system_info(f"Best model checkpoint saved to {best_filepath} with loss {loss:.4f}.")


    def fit(self, train_loader, start_epoch: int = 0):

        for epoch in range(start_epoch, self._epochs):
            self._logger.system_info(f"Epoch {epoch} started.")

            total_loss = 0.0
            n_batches = 0

            for data, labels in train_loader:
                data = data.to(self._device)
                labels = labels.to(self._device)

                outputs = self._model(data)
                loss = self._loss_fn(outputs, labels)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            mean_loss = total_loss / n_batches
            print(f"Epoch {epoch}. Mean Loss: {mean_loss:.4f}")
            
            # check if this is the best model so far
            is_best = mean_loss < self._best_loss
            if is_best:
                self._best_loss = mean_loss
            
            # save checkpoint if needed
            if self._should_save_checkpoint(epoch):
                self._save_checkpoint(epoch, mean_loss, is_best=False)
            
            if is_best and self._save_best:
                self._save_checkpoint(epoch, mean_loss, is_best=True)


        #save_file = export_dir / "model.pth"
        #self._model.cpu()
        #torch.save(self._model.state_dict(), save_file)
        #self._model.to(self._device)

        return self._model

    def load_checkpoint(self, checkpoint_path: Union[Path, str]):
        """Load a checkpoint from the given path.
        
        Args:
            checkpoint_path (Path or str): Path to the checkpoint file.

        Returns:
            Epoch number from the loaded checkpoint 
        
        Example: 
            > trainer = Trainer(model, loss_fn, optimizer, epoch=100)
            > start_epoch = trainer.load_checkpoint("checkpoints/checkpoint_epoch_50.pt")    
            # now you can resume training from epoch 51
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")
        
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint.get('loss', float('inf'))
        self._best_loss = checkpoint.get('best_loss', float('inf'))

        self._logger.system_info(f"Checkpoint loaded from {checkpoint_path}. Resuming from epoch {epoch} with loss {loss:.4f}.")
        return epoch