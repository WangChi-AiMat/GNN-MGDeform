import torch
from torchmetrics import PearsonCorrCoef
from Tool.check import check_tensor_validity


def train(
        model,
        train_loader,
        val_loader,
        normalizer,
        loss_function,
        optimizer,
        device,
):
    model.train()
    train_loss = 0
    train_cnt = 0

    # loss
    for j, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        pred = model(data)
        check_tensor_validity(pred, f"train batch {j} predictions")

        y_norm = normalizer.transform(data.y.view(-1, 1)).to(device)
        loss = loss_function(pred, y_norm)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * pred.size(0)
        train_cnt += pred.size(0)

    avg_train_loss = train_loss / train_cnt

    model.eval()
    train_pearson_metric = PearsonCorrCoef(num_outputs=1).to(device)
    preds_train = []
    targets_train = []

    with torch.no_grad():
        for j, data in enumerate(train_loader):
            data = data.to(device)
            pred = model(data)
            pred_denorm = normalizer.inverse_transform(pred.view(-1, 1))

            train_pearson_metric.update(pred_denorm, data.y.to(device))

            if j == 6:
                preds_train.append(pred_denorm.view(-1).detach().cpu())
                targets_train.append(data.y.view(-1).detach().cpu())

    y_pred_single_train = torch.cat(preds_train, dim=0)
    y_true_single_train = torch.cat(targets_train, dim=0)
    train_pearson = train_pearson_metric.compute().item()

    val_loss = 0.0
    val_cnt = 0
    preds_val = []
    targets_val = []
    val_pearson_metric = PearsonCorrCoef(num_outputs=1).to(device)

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            data = data.to(device)
            pred = model(data)
            check_tensor_validity(pred, f"val batch {i} predictions")

            pred_denorm = normalizer.inverse_transform(pred.view(-1, 1))
            y_norm = normalizer.transform(data.y.view(-1, 1)).to(device)
            y_true = data.y.to(device)

            val_loss += loss_function(pred, y_norm).item() * pred.size(0)
            val_cnt += pred.size(0)
            val_pearson_metric.update(pred_denorm, y_true)

            if i == 0:
                preds_val.append(pred_denorm.view(-1).detach().cpu())
                targets_val.append(y_true.view(-1).detach().cpu())

    y_pred_single_val = torch.cat(preds_val, dim=0)
    y_true_single_val = torch.cat(targets_val, dim=0)

    avg_val_loss = val_loss / val_cnt
    val_pearson = val_pearson_metric.compute().item()

    return {
        "Train Loss": avg_train_loss,
        'Train Pearson': train_pearson,
        "Val Loss": avg_val_loss,
        'Val Pearson': val_pearson,
        'y_pred_train': y_pred_single_train,
        'y_true_train': y_true_single_train,
        'y_pred_val': y_pred_single_val,
        'y_true_val': y_true_single_val,
    }


