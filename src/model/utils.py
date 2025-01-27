import torch.optim as optim

def init_optimizer(model, config):
    assert {"lr", "optimizer_name"}.issubset(config.keys())

    lr = config["lr"]
    optimizer_name = config["optimizer_name"]
    weight_decay = config.get("weight_decay", 0)
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unknown optimizer")

    return optimizer
