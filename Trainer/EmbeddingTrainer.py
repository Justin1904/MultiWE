import torch
import torch.nn

from tqdm import tqdm

def train_epoch(model, optimizer, train_loader, device, clip_grad=0.0, cache_path='./', cache_interval=None):
    train_loss = 0.0
    num_datapoints = 0
    num_iter = 0
    train_iter = tqdm(train_loader)
    losses, t_losses, v_losses, a_losses, rec_losses = [], [], [], [], []
    t_l, v_l, a_l, rec_l = [], [], [], []
    for batch in train_iter:
        num_iter += 1
        model.zero_grad()
        (w, v, a, cw, cv, ca) = batch
        w = w.to(device)
        cw = cw.to(device)
        cv = cv.to(device)
        ca = ca.to(device)
        
        batch_size = w.size(0)
        num_datapoints += batch_size
        loss, details = model(w, cw, cv, ca)
        # loss = model(w, cw, cv, ca).sum()
        train_loss += loss.item()
        loss = loss / batch_size
        loss.backward()
        if clip_grad > 0.0:
            nn.utils.clip_grad_value_([filter(lambda x: x.requires_grad, model.parameters())], clip_grad)
        optimizer.step()

        train_iter.set_description(f"Current batch loss: {round(loss.item(), 4)}")
        # get some training statistics
        loss_t, loss_v, loss_a, loss_rec = details
        t_lambda = model.t_lambda.item()
        v_lambda = model.v_lambda.item()
        a_lambda = model.a_lambda.item()
        rec_lambda = model.rec_lambda.item()

        # collect them into a list
        t_l.append(t_lambda)
        v_l.append(v_lambda)
        a_l.append(a_lambda)
        rec_l.append(rec_lambda)
        t_losses.append(loss_t / batch_size)
        v_losses.append(loss_v / batch_size)
        a_losses.append(loss_a / batch_size)
        rec_losses.append(loss_rec / batch_size)

        # if at certain iter, cache the current model
        if cache_interval is not None and num_iter % cache_interval == 0:
            #torch.save(cache_path+f'model_{num_iter}.pt')
            model.save_embedding(cache_path + f"embedding_{num_iter}.pt")

    return train_loss / num_datapoints,  (t_l, v_l, a_l, rec_l), (t_losses, v_losses, a_losses, rec_losses)


def fit(model, train_loader, optimizer, device, logger, max_epoch=50, patience=10, num_trials=5, clip_grad=0.0, cache_path='./', cache_interval=2000):
    t_losses, v_losses, a_losses, rec_losses = [], [], [], []
    t_l, v_l, a_l, rec_l = [], [], [], []
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    lr_scheduler.step()
    best_loss = float('inf')
    curr_patience = patience
    for e in range(max_epoch):
        train_loss, details, weights = train_epoch(model, optimizer, train_loader, device, clip_grad, cache_path=cache_path, cache_interval=cache_interval)
        t_losses.extend(details[0])
        v_losses.extend(details[1])
        a_losses.extend(details[2])
        rec_losses.extend(details[3])

        t_l.extend(weights[0])
        v_l.extend(weights[1])
        a_l.extend(weights[2])
        rec_l.extend(weights[3]) 

        logger.log(f"Epoch {e+1}/{max_epoch}, training loss: {train_loss}, current patience: {curr_patience}, current trial: {num_trials}")
        if train_loss <= best_loss:
            logger.log("Found new best model, saving to disk.")
            best_loss = train_loss
            torch.save(model.state_dict(), cache_path+'model.pt')
            torch.save(optimizer.state_dict(), cache_path+'optim.pt')
            curr_patience = patience
        else:
            curr_patience -= 1
            if curr_patience < 0:
                logger.log("Running out of patience, loading previous best model.")
                num_trials -= 1
                model.load_state_dict(torch.load(cache_path+'model.pt'))
                optimizer.load_state_dict(torch.load(cache_path+'optim.pt'))
                model.save_embedding(cache_path+f'embedding_trial_{num_trials}.txt')
                lr_scheduler.step()
                curr_lr = optimizer.state_dict()['param_groups'][0]['lr']
                logger.log(f"Current learning rate: {curr_lr}")

        if num_trials <= 0:
            logger.log("Running out of patience, early stopping.")
            break
    return (t_losses, v_losses, a_losses, rec_losses), (t_l, v_l, a_l, rec_l)

