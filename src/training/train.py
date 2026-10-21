"""Interactive single-model training loop with live loss plots.

Intended for notebook exploration of one model. For batch training of model
ensembles (no plotting, sample-weighted metrics, auto-saving), use
``training.mass_training.train_many_mlps_on_dataset_instance`` instead.
"""

import torch
import torch.nn.functional as F

from training.plot_training import plot_losses

from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mlp_train_step(X, y, mlp_model, mlp_optimizer):
    mlp_model.train()

    y = y.float()
    pred = mlp_model(X).view(-1)
    loss = F.binary_cross_entropy_with_logits(pred, y, reduction='mean')

    mlp_optimizer.zero_grad()
    loss.backward()
    mlp_optimizer.step()
    return loss.item()

@torch.no_grad()
def mlp_test_step(X, y, mlp_model):
    mlp_model.eval()

    y = y.float()
    pred = mlp_model(X).view(-1)

    loss = F.binary_cross_entropy_with_logits(pred, y, reduction="mean")

    pred_labels = (torch.sigmoid(pred) > 0.5).long()
    acc = (pred_labels == y.long()).float().mean().item()

    return loss.item(), acc

@torch.no_grad()
def evaluate(model, test_loader, device):
    test_losses = []
    test_accs = []

    for X, y in test_loader:
        X, y = X.to(device), y.to(device)

        loss, acc = mlp_test_step(X, y, model)

        test_losses.append(loss)
        test_accs.append(acc)

    return sum(test_losses) / len(test_losses), sum(test_accs) / len(test_accs)

def train_mlp(
    model,
    train_loader,
    test_loader,
    optimizer,
    n_epochs,
    visualize_step=10,
    device=DEVICE,
    target_test_acc=0.999,
    patience=5,
    min_epochs=20,
):
    train_losses = []
    test_losses = []
    test_accs = []

    model.to(device)

    good_epochs = 0

    for epoch in tqdm(range(n_epochs)):
        batch_train_losses = []

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            train_loss = mlp_train_step(X, y, model, optimizer)
            batch_train_losses.append(train_loss)

        epoch_train_loss = sum(batch_train_losses) / len(batch_train_losses)
        epoch_test_loss, epoch_test_acc = evaluate(model, test_loader, device)

        train_losses.append(epoch_train_loss)
        test_losses.append(epoch_test_loss)
        test_accs.append(epoch_test_acc)

        if epoch % visualize_step == 0 or epoch == n_epochs - 1:
            plot_losses(train_losses, test_losses, test_accs)
            print(
                f"epoch {epoch:04d} | "
                f"train loss: {epoch_train_loss:.4f} | "
                f"test loss: {epoch_test_loss:.4f} | "
                f"test acc: {epoch_test_acc:.4f}"
            )

        if epoch >= min_epochs and epoch_test_acc >= target_test_acc:
            good_epochs += 1
        else:
            good_epochs = 0

        if good_epochs >= patience:
            print(
                f"Early stopping at epoch {epoch:04d}: "
                f"test acc >= {target_test_acc:.4f} "
                f"for {patience} consecutive epochs."
            )
            break

    return train_losses, test_losses, test_accs