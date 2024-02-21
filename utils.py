import torch
import wandb
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

def train_test(model, train_dataloader, test_dataloader, loss_fn, optimiser, config, device):
    wandb.watch(model, loss_fn, log="all", log_freq=10)

    train_example_ct=0
    train_batch_ct=0
    test_example_ct = 0
    test_batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        
        for _, (X, y) in enumerate(train_dataloader):
            model.train()
            loss, acc = train_batch(X, y, model, loss_fn, optimiser, device)
            train_example_ct += len(X)
            train_batch_ct += 1

            if train_batch_ct % 500 == 0:
                model.eval()
                with torch.inference_mode():
                    test_acc = 0
                    for _, (X, y) in enumerate(test_dataloader):
                        test_batch_ct += 1
                        test_example_ct += len(X)

                        X, y = X.to(device), y.to(device)
                        y_logits = model(X)
                        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
                        test_acc += (y_pred==y).sum().item()/len(y)
                test_acc /= len(test_dataloader)
                log_train_test(loss, acc, test_acc, train_example_ct, epoch)
        
       

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

def log_train_test(loss, accuracy, test_accuracy, example_count, epoch):
    loss = float(loss)

    wandb.log({"epoch": epoch, "loss": loss, "accuracy": accuracy, "test_accuracy": test_accuracy}, step=example_count)
    print(f"Train Loss After {str(example_count).zfill(5)} examples: {loss:.3f}")

def make(model, train_data, test_data, config, device):
    train_dataloader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True
    )

    test_dataloader = DataLoader(
        test_data, batch_size=config.batch_size, shuffle=False
    )

    model = model.to(device)
    print(model)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        params=model.parameters(), lr=config.learning_rate
    )
    return model, train_dataloader, test_dataloader, loss_fn, optimiser

def model_pipeline(proj: str, model: torch.nn.Module, train_data, test_data, config, device):
    with wandb.init(project=proj, config=config):
        config = wandb.config
        model, train_dataloader, test_dataloader, loss_fn, optimiser = make(
            model, train_data, test_data, config, device
        )

        train_test(model, train_dataloader, test_dataloader, loss_fn, optimiser, config, device)
    return model

