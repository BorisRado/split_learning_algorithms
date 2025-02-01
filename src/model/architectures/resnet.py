import torch
import torch.nn as nn
import torchvision.models as models


def resnet18(pretrained, num_classes, partition, last_client_layer):
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    client_layers = []
    server_layers = []
    server = False
    for k, _ in model.named_children():
        if not server:
            client_layers.append(k)
            if k == last_client_layer:
                server=True
        else:
            server_layers.append(k)

    if partition == "client":
        model = nn.Sequential(*[getattr(model, k) for k in client_layers])
        if last_client_layer == "fc":
            model = model.insert(-1, nn.Flatten())
    elif partition == "server":
        model = nn.Sequential(*[getattr(model, k) for k in server_layers])
        model = model.insert(-1, nn.Flatten())
    elif partition == "server_encoder":
        model = nn.Sequential(*[getattr(model, k) for k in server_layers[:-1]])
        model = model.append(nn.Flatten())
    elif partition == "final_clf_head":
        model = model.fc
    elif partition == "intermediate_clf_head":
        assert last_client_layer == "layer1"
        model = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    elif partition == "decoder":
        # for LocFedMix
        model = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 3, 3, 2, 1, 1),
            nn.Sigmoid()
        )

    else:
        assert partition is None and last_client_layer is None

    model.is_complete_model = partition is None or \
        (partition == "client" and last_client_layer == "fc")

    return model


if __name__ == "__main__":
    from src.utils.stochasticity import TempRng, set_seed

    seed = 1602
    common_kwargs = {"pretrained": False, "num_classes": 10}
    with TempRng(seed=seed):
        client_model = resnet18(partition="client", last_client_layer="maxpool", **common_kwargs)

    with TempRng(seed=seed):
        server_model = resnet18(partition="server", last_client_layer="maxpool", **common_kwargs)

    with TempRng(seed=seed):
        server_encoder = resnet18(partition="server_encoder", last_client_layer="maxpool", **common_kwargs)

    with TempRng(seed=seed):
        client_clf_head = resnet18(partition="final_clf_head", last_client_layer="maxpool", **common_kwargs)

    with TempRng(seed=seed):
        whole_model = resnet18(partition=None, last_client_layer=None, **common_kwargs)

    for _ in range(10):
        x = torch.rand(10, 3, 224, 224)
        set_seed(seed=seed)
        whole_model_preds = whole_model(x)
        set_seed(seed=seed)
        assert server_model(client_model(x)).allclose(whole_model_preds)

        set_seed(seed=seed)
        assert client_clf_head(server_encoder(client_model(x))).allclose(whole_model_preds)
