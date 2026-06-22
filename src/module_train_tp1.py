# Core Python libraries
import os
import tqdm

# PyTorch libraries
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import precision_score, recall_score, f1_score

import matplotlib.pyplot as plt


def train_test(train_iter:torch.utils.data.DataLoader,
               eval_iter:torch.utils.data.DataLoader,
               model:torch.nn, 
               loss_function:torch.nn.modules.loss,
               optimizer:torch.optim.Optimizer, 
               epochs:int,  
               log_dir:str,
               checkpoint_path:str,
               device:str,
               early_stopping_patience=15, 
               reduce_lr_patience=5,
               reduce_lr_factor=0.1
    ):
    
    
    # Initialisation
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  patience=reduce_lr_patience,
                                  factor=reduce_lr_factor)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    #Variable d'état qui vont évoluer avec l'entrainement
    best_loss = float('inf')
    patience_counter = 0
    current_lr = 0

    list_train_loss, list_eval_loss = [], []
    list_train_accuracy, list_eval_accuracy = [], []
    list_train_f1, list_eval_f1 = [], []

    model.to(device)

    # Iterate over the training data for the specified number of epochs
    for epoch in tqdm.tqdm(range(epochs)):

        model.train()

        total_train_loss = 0.0
        total_train_samples = 0
        correct_predictions_train = 0

        predicted_labels_train_list = []
        targets_train_list = []

        for batch in tqdm.tqdm(train_iter):

            optimizer.zero_grad() #réinitialise les gradients 

            input_ids = batch['input_ids'].to(device)
            targets = batch["labels"].to(device)

            outputs = model(input_ids)
            loss = loss_function(outputs, targets)

            loss.backward() #effectue la back propagation
            optimizer.step() #ajoute une étape de descente de gradient sur les poids du modèle

            total_train_loss += loss.item() * len(input_ids) #pondère la loss avec le nombre d'individus dans le batch
            total_train_samples += len(input_ids)

            predicted_labels = torch.argmax(outputs, dim=1)

            correct_predictions_train += (predicted_labels == targets).sum().item()

            predicted_labels_train_list.extend(predicted_labels.tolist())
            targets_train_list.extend(targets.tolist())

        
        # Evaluate on the validation set after every epoch
        model.eval()
        total_eval_loss = 0.0
        total_eval_samples = 0

        correct_predictions_eval = 0

        predicted_labels_eval_list = []
        targets_eval_list = []

        with torch.no_grad():
            for batch in tqdm.tqdm(eval_iter):

                input_ids = batch['input_ids'].to(device)
                targets = batch["labels"].to(device)

                outputs = model(input_ids).to(device)
                eval_loss = loss_function(outputs, targets)

                total_eval_loss += eval_loss.item() * len(input_ids)
                total_eval_samples += len(input_ids)

                predicted_labels = torch.argmax(outputs, dim=1)

                correct_predictions_eval += (predicted_labels == targets).sum().item()

                predicted_labels_eval_list.extend(predicted_labels.tolist())
                targets_eval_list.extend(targets.tolist())

        #Loss                
        writer.add_scalars("loss",{'train':total_train_loss/total_train_samples,
                                   'eval':total_eval_loss/total_eval_samples}, epoch+1)
        list_train_loss.append(total_train_loss/total_train_samples)
        list_eval_loss.append(total_eval_loss/total_eval_samples)

        #accuracy
        writer.add_scalars("accuracy",{'train':correct_predictions_train/total_train_samples,
                                   'eval':correct_predictions_eval/total_eval_samples}, epoch+1)
        list_train_accuracy.append(correct_predictions_train / total_train_samples)
        list_eval_accuracy.append(correct_predictions_eval / total_eval_samples)

        #precision
        writer.add_scalars("precision",{'train':precision_score(targets_train_list, predicted_labels_train_list),
                                   'eval':precision_score(targets_eval_list, predicted_labels_eval_list)}, epoch+1)
        
        #recall
        writer.add_scalars("recall",{'train':recall_score(targets_train_list, predicted_labels_train_list),
                                   'eval':recall_score(targets_eval_list, predicted_labels_eval_list)}, epoch+1)
        
        #F1_score
        writer.add_scalars("f1_score",{'train':f1_score(targets_train_list, predicted_labels_train_list),
                                   'eval':f1_score(targets_eval_list, predicted_labels_eval_list)}, epoch+1)
        list_train_f1.append(f1_score(targets_train_list, predicted_labels_train_list))
        list_eval_f1.append(f1_score(targets_eval_list, predicted_labels_eval_list))
        

        # Enregistrer le taux d'apprentissage actuel
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            writer.add_scalar("lr", current_lr, epoch)

        # Réduit automatiquement le taux d'apprentissage si la perte de validation ne diminue pas sous les critères passés
        scheduler.step(list_eval_loss[-1])

        # Early Stopping
        if list_eval_loss[-1] < best_loss:
            best_loss = list_eval_loss[-1]
            patience_counter = 0
            # Sauvegarder le meilleur modèle
            torch.save(model.state_dict(), checkpoint_path)
            epoch_best = epoch+1
            print(f'Find a better model at the epoch {epoch+1} - Train Loss: {list_train_loss[-1]:.4f}, eval Loss: {list_eval_loss[-1]:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                model.load_state_dict(torch.load(checkpoint_path))
                print("current_lr : ", current_lr)
                print("epoch_best : ", epoch_best)
                break

    writer.close()

    return list_train_loss, list_eval_loss ,list_train_accuracy, list_eval_accuracy ,list_train_f1, list_eval_f1


def plot_metrics(train_metrics, test_metrics, metric_name):
    """
    Plots train and test metrics over epochs using Matplotlib.

    Args:
        train_metrics (list): List of training metric values.
        test_metrics (list): List of testing metric values.
        metric_name (str): Name of the metric to display on the plot.
    """
    epochs = list(range(1, len(train_metrics) + 1))  # Create a list of epoch numbers

    # Create the figure
    plt.figure(figsize=(8, 6))

    # Plot Train metric line
    plt.plot(epochs, train_metrics, label=f'Train {metric_name}', marker='o', linestyle='-', linewidth=2)

    # Plot Test metric line
    plt.plot(epochs, test_metrics, label=f'Test {metric_name}', marker='x', linestyle='--', linewidth=2)

    # Adding titles and labels
    plt.title(f'{metric_name} Over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    
    # Display the legend
    plt.legend()

    # Show grid for better readability
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()
