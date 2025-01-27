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
