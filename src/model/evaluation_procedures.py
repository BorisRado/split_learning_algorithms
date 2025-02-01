import torch
import torch.nn.functional as F


def evaluate_model(model, server_model_proxy, valloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    tot_loss, corrects = 0., 0
    for batch in valloader:
        img, labels = batch["img"].to(device), batch["label"]

        with torch.no_grad():
            if isinstance(model, torch.nn.ModuleDict):
                output = model["encoder"](img)
            else:
                output = model(img)

        if model.is_complete_model:
            logits = output.cpu()
        else:
            logits = server_model_proxy.get_logits(embeddings=output,)

        tot_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        corrects += (logits.argmax(dim=1) == labels).sum().item()

    return {
        "loss": tot_loss / len(valloader.dataset),
        "accuracy": corrects / len(valloader.dataset),
    }


def evaluate_client_and_server_clf_head(model, server_model_proxy, valloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    client_loss, client_corrects = 0., 0
    server_loss, server_corrects = 0., 0

    for batch in valloader:
        img, labels = batch["img"].to(device), batch["label"]

        with torch.no_grad():
            client_embs = model["encoder"](img)
            client_logits = model["clf_head"](client_embs)
            server_logits = server_model_proxy.get_logits(embeddings=client_embs).to(device)

        client_loss += F.cross_entropy(client_logits, labels, reduction="sum").item()
        client_corrects += (client_logits.argmax(dim=1) == labels).sum().item()
        server_loss += F.cross_entropy(server_logits, labels, reduction="sum").item()
        server_corrects += (server_logits.argmax(dim=1) == labels).sum().item()

    return {
        "client_loss": client_loss / len(valloader.dataset),
        "client_accuracy": client_corrects / len(valloader.dataset),
        "loss": server_loss / len(valloader.dataset),
        "accuracy": server_corrects / len(valloader.dataset),
    }


def evaluate_ushaped(model, server_model_proxy, valloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    loss, corrects = 0., 0
    for batch in valloader:
        img, labels = batch["img"].to(device), batch["label"]

        with torch.no_grad():
            client_embs = model["encoder"](img)

        server_embs = server_model_proxy.u_forward_inference(embeddings=client_embs).to(device)

        with torch.no_grad():
            client_logits = model["clf_head"](server_embs)

        loss += F.cross_entropy(client_logits, labels, reduction="sum").item()
        corrects += (client_logits.argmax(dim=1) == labels).sum().item()

    return {
        "loss": loss / len(valloader.dataset),
        "accuracy": corrects / len(valloader.dataset),
    }
