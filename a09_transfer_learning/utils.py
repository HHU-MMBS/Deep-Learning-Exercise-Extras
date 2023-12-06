from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt


def imshow(img, i=0, mean=torch.tensor([0.0], dtype=torch.float32), std=torch.tensor([1], dtype=torch.float32)):
    """
    shows an image on the screen. mean of 0 and variance of 1 will show the images unchanged in the screen
    """
    # undoes the normalization
    unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    plt.subplot(1, 10 ,i+1)
    npimg = unnormalize(img).numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total = 0, 0
    loss_step = []
    for inp_data, labels in val_loader:
        # Move the data to the GPU
        labels = labels.view(labels.shape[0]).to(device)
        inp_data = inp_data.to(device)

        outputs = model(inp_data)
        val_loss = criterion(outputs, labels)
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        loss_step.append(val_loss.item())
    # don't forget to take the means here
    val_acc = (100 * correct / total).cpu().item()
    val_loss_epoch = torch.tensor(loss_step).mean().item()
    return val_acc, val_loss_epoch


def train_one_epoch(model, optimizer, train_loader, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_step = []
    correct, total = 0, 0
    for (inp_data, labels) in train_loader:
        # Move the data to the GPU
        labels = labels.view(labels.shape[0]).to(device)
        inp_data = inp_data.to(device)
        outputs = model(inp_data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            loss_step.append(loss.item())
    # don't forget the means here
    loss_curr_epoch = np.mean(loss_step)
    train_acc = (100 * correct / total).cpu()
    return loss_curr_epoch, train_acc


def train(model, optimizer, num_epochs, train_loader, val_loader, device):
    best_val_loss = float('inf')
    best_val_acc = 0
    model = model.to(device)
    dict_log = {"train_acc_epoch": [], "val_acc_epoch": [], "loss_epoch": [], "val_loss": []}
    train_acc, _ = validate(model, train_loader, device)
    val_acc, _ = validate(model, val_loader, device)
    print(f'Init Accuracy of the model: Train:{train_acc:.3f} \t Val:{val_acc:3f}')
    with tqdm(range(num_epochs)) as pbar:
        for epoch in pbar:
            loss_curr_epoch, train_acc = train_one_epoch(model, optimizer, train_loader, device)
            val_acc, val_loss = validate(model, val_loader, device)

            # Print epoch results to screen
            msg = f'Ep {epoch}/{num_epochs}: Accuracy : Train:{train_acc:.2f} \t Val:{val_acc:.2f} || Loss: Train {loss_curr_epoch:.3f} \t Val {val_loss:.3f}'
            pbar.set_description(msg)
            # Track stats
            dict_log["train_acc_epoch"].append(train_acc)
            dict_log["val_acc_epoch"].append(val_acc)
            dict_log["loss_epoch"].append(loss_curr_epoch)
            dict_log["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                      'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'loss': val_loss,
                      }, f'best_model_min_val_loss.pth')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                      'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'loss': val_loss,
                      }, f'best_model_max_val_acc.pth')
    return dict_log


def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model {path} is loaded from epoch {checkpoint['epoch']} , loss {checkpoint['loss']}")
    return model


def test_model(model, path, test_loader, device='cuda'):
    model = load_model(model, path)
    model.to(device)
    model.eval()
    return validate(model, test_loader, device)


def plot_stats(dict_log, modelname="", baseline=90, title=None):
    fontsize = 14
    plt.subplots_adjust(hspace=0.3)
    plt.subplot(2, 1, 1)
    x_axis = list(range(len(dict_log["val_acc_epoch"])))
    plt.plot(dict_log["train_acc_epoch"], label=f'{modelname} Train accuracy')
    plt.scatter(x_axis, dict_log["train_acc_epoch"])

    plt.plot(dict_log["val_acc_epoch"], label=f'{modelname} Validation accuracy')
    plt.scatter(x_axis, dict_log["val_acc_epoch"])

    plt.ylabel('Accuracy in %')
    plt.xlabel('Number of Epochs')
    plt.title("Accuracy over epochs", fontsize=fontsize)
    plt.axhline(y=baseline, color='red', label="Acceptable accuracy")
    plt.legend(fontsize=fontsize)

    plt.subplot(2,1,2)
    plt.plot(dict_log["loss_epoch"] , label="Training")

    plt.scatter(x_axis, dict_log["loss_epoch"], )
    plt.plot(dict_log["val_loss"] , label='Validation')
    plt.scatter(x_axis, dict_log["val_loss"])

    plt.ylabel('Loss value')
    plt.xlabel('Number of Epochs')
    plt.title("Loss over epochs", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    if title is not None:
        plt.savefig(title)
