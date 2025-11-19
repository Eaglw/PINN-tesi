import matplotlib.pyplot as plt
import os

class TrainingHistory:
    """
    Una classe per registrare e visualizzare l'andamento delle loss durante il training.
    """
    def __init__(self):
        self.epochs = []
        self.losses = {}

    def update(self, epoch, loss_dict):
        """
        Registra i valori delle loss per un dato'epoch'.

        Args:
            epoch (int): L'epoch corrente.
            loss_dict (dict): Un dizionario con i nomi delle loss e i loro valori.
                              Es: {'total_loss': 1.5, 'pde_loss': 1.2, ...}
        """
        self.epochs.append(epoch)
        for name, value in loss_dict.items():
            # Se è un tensore, estrai il valore numerico
            val = value.item() if hasattr(value, 'item') else value
            
            if name not in self.losses:
                self.losses[name] = []
            self.losses[name].append(val)

    def plot_losses(self, save_path=None, experiment_name=""):
        """
        Genera un grafico con l'andamento di tutte le loss registrate.
        """
        plt.figure(figsize=(8, 4))
        ax = plt.gca()
        for name, values in self.losses.items():
            if name == "total_loss":
                plt.plot(self.epochs, values, linewidth=4, label=name)
            else:
                plt.plot(self.epochs, values, label=name)
        
        plt.title(f'Andamento Loss - {experiment_name}', y=0.85)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        # Stile coerente con plot_result
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
        plt.setp(l.get_texts(), color="k")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Grafico delle loss salvato in: {save_path}")
        
        plt.close() # Chiude la figura per liberare memoria
