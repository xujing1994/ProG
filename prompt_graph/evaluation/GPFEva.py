import torchmetrics
import torch
from tqdm import tqdm
import ipdb
def GPFEva(loader, gnn, prompt, answering, num_class, device):
    prompt.eval()
    if answering:
        answering.eval()
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    accuracy.reset()
    macro_f1.reset()
    auroc.reset()
    auprc.reset()
    outs = []
    labels = []
    acc = 0.0
    with torch.no_grad(): 
        for batch_id, batch in enumerate(loader): 
            labels.append(batch.y)
            batch = batch.to(device) 
            batch.x = prompt.add(batch.x)
            out = gnn(batch.x, batch.edge_index, batch.batch)
            if answering:
                out = answering(out)  
            # out: posterior probabilities, used to train the attack model
            pred = out.argmax(dim=1)  
            loss = criterion(out, batch.y)
            # ipdb.set_trace()
            acc += accuracy(pred, batch.y).item()
            ma_f1 = macro_f1(pred, batch.y)
            roc = auroc(out, batch.y)
            prc = auprc(out, batch.y)
            # if len(loader) > 20:
            #     print("Batch {}/{} Acc: {:.4f} | Macro-F1: {:.4f}| AUROC: {:.4f}| AUPRC: {:.4f}".format(batch_id,len(loader), acc.item(), ma_f1.item(),roc.item(), prc.item()))
            outs.append(out)
            # print(acc)
    outs_ = torch.cat(outs, dim=0)
    #acc = accuracy.compute()
    ma_f1 = macro_f1.compute()
    roc = auroc.compute()
    prc = auprc.compute()
       
    return acc / len(loader), ma_f1.item(), roc.item(),prc.item(), torch.cat(outs, dim=0), loss.item(), torch.cat(labels, dim=0)