import torch
import torch.nn.functional as F

from slbd.common import RequestType


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
    else:
        tot_loss /= len(trainloader)

    return {"train_loss": tot_loss}


def train_u_shaped(model, server_model_proxy, trainloader, optimizer):
    """U-shaped architecture"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    model.to(device)

    tot_loss = 0.
    for batch in trainloader:
        img, labels = batch["img"].to(device), batch["label"].to(device)

        client_embs = model["encoder"](img)

        server_embs = server_model_proxy.u_forward(embeddings=client_embs).to(device)
        server_embs.requires_grad_(True)

        client_preds = model["clf_head"](server_embs)

        optimizer.zero_grad()

        loss = F.cross_entropy(client_preds, labels)
        loss.backward()
        server_grad = server_model_proxy.u_backward(gradient=server_embs.grad).to(device)
        client_embs.backward(server_grad)
        optimizer.step()

        tot_loss += loss.item()
    tot_loss /= len(trainloader)

    return {"train_loss": tot_loss}


def train_fsl(model, server_model_proxy, trainloader, optimizer):
    """Implementation of method proposed in
    `Accelerating Federated Split Learning via Local-Loss-Based Training`
    Client has a classification head used to update its model
    Clients sends embeddings and target labels to the server which updates its model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    model.to(device)

    tot_loss = 0.
    for batch in trainloader:
        img, labels = batch["img"].to(device), batch["label"].to(device)

        client_embs = model["encoder"](img)

        server_model_proxy.update_server_model(
            embeddings=client_embs,
            labels=labels,
            _type_=RequestType.STREAM,
        )

        client_preds = model["clf_head"](client_embs)

        optimizer.zero_grad()
        loss = F.cross_entropy(client_preds, labels)
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
    tot_loss /= len(trainloader)
    server_loss = server_model_proxy.get_round_loss().item()

    return {
        "train_loss": tot_loss,
        "server_train_loss": server_loss,
    }


def train_streamsl(model, server_model_proxy, trainloader, optimizer):
    """Forward pass on client model, send embeddings and target labels
    to the server, which updates the server model. Equivalent to feezing
    the client layers and training the server model only.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    model.to(device)
    model.requires_grad_(False)

    tot_loss = 0.
    for batch in trainloader:
        img, labels = batch["img"].to(device), batch["label"]

        with torch.no_grad():
            client_embs = model(img)

        server_model_proxy.update_server_model(
            embeddings=client_embs,
            labels=labels,
            _type_=RequestType.STREAM,
        )

    tot_loss = server_model_proxy.get_round_loss().item()

    return { "train_loss": tot_loss, }


def train_locfedmix(model, server_model_proxy, trainloader, optimizer):
    """Implementation of the LocFedMix algorithm proposed in
        LocFedMix-SL: Localize, Federate, and Mix for Improved Scalability,
        Convergence, and Latency in Split Learning
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    model.to(device)
    tot_recon_loss = 0.

    for batch in trainloader:
        img, labels = batch["img"].to(device), batch["label"]

        client_embs = model["encoder"](img).detach()
        gradient_future = server_model_proxy.locfedmix_gradient(
            embeddings=client_embs,
            labels=labels,
            _type_=RequestType.FUTURE,
        )

        client_embs.requires_grad_(True)
        recon_images = model["decoder"](client_embs)
        infopro_loss = F.mse_loss(recon_images, img)

        optimizer.zero_grad()
        infopro_loss.backward()
        server_gradient = gradient_future.get_response().to(device)
        gradient = server_gradient + client_embs.grad
        client_embs.backward(gradient)
        optimizer.step()

        tot_recon_loss += infopro_loss.item()

    tot_recon_loss /= len(trainloader)
    return {
        "recon_loss": tot_recon_loss,
        "train_loss": server_model_proxy.get_round_loss().item(),
    }


def train_splitavg(model, server_model_proxy, trainloader, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    model.to(device)

    tot_loss = 0.
    for batch in trainloader:
        img, labels = batch["img"], batch["label"]
        img = img.to(device)

        output = model(img)

        optimizer.zero_grad()

        assert not model.is_complete_model
        gradient = server_model_proxy.serve_splitavg_gradient(
            embeddings=output,
            labels=labels
        ).to(device)
        output.backward(gradient)

        optimizer.step()
    if not model.is_complete_model:
        tot_loss = server_model_proxy.get_round_loss().item()
    else:
        tot_loss /= len(trainloader)

    return {"train_loss": tot_loss}
