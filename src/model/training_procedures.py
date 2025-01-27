import torch
import torch.nn as nn
import torch.nn.functional as F


def train_ce(model, server_model_proxy, trainloader, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    model.to(device)

    tot_loss = 0.
    for batch in trainloader:
        img, labels = batch["img"], batch["label"]
        img = img.to(device)

        output = model(img)

        optimizer.zero_grad()

        if model.is_complete_model:
            loss = F.cross_entropy(output, labels.to(device))
            loss.backward()
            tot_loss += loss.item()
        else:
            gradient = server_model_proxy.serve_grad_request(
                embeddings=output,
                labels=labels
            ).to(device)
            output.backward(gradient)

        optimizer.step()
    if not model.is_complete_model:
        tot_loss = server_model_proxy.get_round_loss().item()

    tot_loss /= len(trainloader)

    return tot_loss
