# %%
import os
import torch.nn as nn
import torch
import torch.nn.functional as F

# %%


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.l1 = nn.Linear(29, 22)
        self.l2 = nn.Linear(22, 15)
        self.l3 = nn.Linear(15, 10)
        self.l4 = nn.Linear(10, 22)
        self.l5 = nn.Linear(22, 29)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(29, 22)
        self.l2 = nn.Linear(22, 15)
        self.l3 = nn.Linear(15, 10)
        self.l4 = nn.Linear(10, 5)
        self.l5 = nn.Linear(5, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        # x = F.log_softmax(self.l5(x), dim=1)
        # x = F.relu(self.l5(x))
        x = torch.sigmoid(self.l5(x))
        return x


def get_state_path(
    auto_encoder, classifier, percent,
    auto_encoder_criterion, classifier_criterion,
    auto_encoder_optimizer, classifier_optimizer
):
    path = "models"
    net_name = f"{auto_encoder._get_name()}{classifier._get_name()}-{int(percent*100)}-sigmoid"
    criterion_name = f"{auto_encoder_criterion.__class__.__name__}{classifier_criterion.__class__.__name__}"
    optimizer_name = f"{auto_encoder_optimizer.__class__.__name__}{classifier_optimizer.__class__.__name__}"
    model_config_name = f"{net_name}-optimizer-{optimizer_name}-loss-{criterion_name}"
    os.makedirs(path, exist_ok=True)
    state_file_name = f"{path}/state-{model_config_name}.pth"
    return model_config_name, state_file_name
