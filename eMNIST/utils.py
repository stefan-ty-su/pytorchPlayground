import torch
import wandb
from tqdm.auto import tqdm

def train(model, train_dataloader, loss_fn, optimiser, config, device):
    wandb.watch(model, loss_fn, log="all", log_freq=10)

    model.train()
    example_ct=0
    batch_ct=0
    for epoch in tqdm(range(config.epochs)):
        for _, (X, y) in enumerate(train_dataloader):

            loss, acc = train_batch(X, y, model, loss_fn, optimiser, device)
            example_ct += len(X)
            batch_ct += 1

            if ((batch_ct) % 500) == 0:
                train_log(loss, acc, example_ct, epoch)

def train_batch(X, y, model, loss_fn, optimiser, device):
    X, y = X.to(device), y.to(device)

    y_logits = model(X)
    loss = loss_fn(y_logits, y)

    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    acc = (y_pred==y).sum().item()/len(y)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    return loss, acc

def train_log(loss, accuracy, example_count, epoch):
    loss = float(loss)

    wandb.log({"epoch": epoch, "loss": loss, "accuracy": accuracy}, step=example_count)
    print(f"Loss After {str(example_count).zfill(5)} examples: {loss:.3f}")

def test(model, dataloader, device):
    model.eval()
    with torch.inference_mode():
        train_acc = 0
        for _, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
            train_acc += (y_pred==y).sum().item()/len(y)

        train_acc /= len(dataloader)
        wandb.log({"test_accuracy": train_acc})