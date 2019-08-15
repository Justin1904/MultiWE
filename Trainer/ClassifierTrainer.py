import torch
import torch.nn as nn

from tqdm import tqdm

def train_epoch(model, optimizer, train_loader, criterion, device, clip_grad=0.0):
    train_loss = 0.0
    num_datapoints = 0
    train_iter = tqdm(train_loader)
    losses = []
    for batch in train_iter:
        model.zero_grad()
        t, v, a, l, y = batch
        t = t.to(device)
        v = v.to(device)
        a = a.to(device)
        l = l.to(device)
        y = y.to(device)

        batch_size = l.size(0)
        num_datapoints += batch_size
        y_tilde = model(t, v, a, l)
        loss = criterion(y_tilde, y)
        loss = loss / batch_size
        if clip_grad > 0.0:
            nn.utils.clip_grad_value_([filter(lambda x: x.requires_grad, model.parameters())])
        optimizer.step()

        train_iter.set_description(f"Current batch loss: {round(loss.item(), 4)}")
        losses.append(loss.item())


@torch.no_grad()
def validate(model, valid_loader, metric, device):
    y_pred = []
    y_true = []
    for batch in valid_loader:
        t, v, a, l, y = batch
        t = t.to(device)
        v = v.to(device)
        l = l.to(device)
        y = y.to('cpu').detach().numpy()
        y_tilde = model(t, a, v, l)
        y_tilde = y_tilde.to('cpu').detach().numpy()
        y_pred.append(y_tilde)
        y_true.append(y)
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    metric_value = metric(y_true, y_pred)
    return metric_value


@torch.no_grad()
def evaluate(model, test_loader, metric, device):
    y_pred = []
    y_true = []
    for batch in valid_loader:
        t, v, a, l, y = batch
        t = t.to(device)
        v = v.to(device)
        l = l.to(device)
        y = y.to('cpu').detach().numpy()
        y_tilde = model(t, a, v, l)
        y_tilde = y_tilde.to('cpu').detach().numpy()
        y_pred.append(y_tilde)
        y_true.append(y)
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    return y_pred, y_true


