# %%
from train import TrainModel
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data import get_data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def draw_loss_curve(current_epoch, name, res):
    plt.clf()
    x_epoch = list(range(1, current_epoch+1))
    loss_train = res["loss_train"]
    loss_val = res["loss_val"]
    plt.plot(x_epoch, loss_train, 'bo-', label='train')
    plt.plot(x_epoch, loss_val, 'ro-', label='val')
    plt.legend()
    os.makedirs("loss_graphs", exist_ok=True)
    plt.savefig(os.path.join('./loss_graphs', f'loss_{name}.jpg'))
    plt.clf()

    plt.plot(x_epoch, loss_train, 'bo-', label='train')
    plt.legend()
    os.makedirs("loss_graphs", exist_ok=True)
    plt.savefig(os.path.join('./loss_graphs', f'loss_train_{name}.jpg'))
    plt.clf()

    plt.plot(x_epoch, loss_val, 'ro-', label='val')
    plt.legend()
    os.makedirs("loss_graphs", exist_ok=True)
    plt.savefig(os.path.join('./loss_graphs', f'loss_val_{name}.jpg'))
    plt.clf()



def draw_acc_curve(current_epoch, name, res):
    plt.clf()
    x_epoch = list(range(1, current_epoch+1))
    acc_train = res["acc_train"]
    acc_val = res["acc_val"]
    plt.plot(x_epoch, acc_train, 'bo-', label='train')
    plt.plot(x_epoch, acc_val, 'ro-', label='val')
    plt.legend()
    os.makedirs("loss_graphs", exist_ok=True)
    plt.savefig(os.path.join('./loss_graphs', f'acc_{name}.jpg'))
    plt.clf()

    plt.plot(x_epoch, acc_train, 'bo-', label='train')
    plt.legend()
    os.makedirs("loss_graphs", exist_ok=True)
    plt.savefig(os.path.join('./loss_graphs', f'acc_train_{name}.jpg'))
    plt.clf()

    plt.plot(x_epoch, acc_val, 'ro-', label='val')
    plt.legend()
    os.makedirs("loss_graphs", exist_ok=True)
    plt.savefig(os.path.join('./loss_graphs', f'acc_val_{name}.jpg'))
    plt.clf()

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
        x = F.log_softmax(self.l5(x), dim=1)
        # x = F.relu(self.l5(x))
        return x


data_loader, dataset, dataset_sizes = get_data(over_sample=True, percent=0.8)

auto_encoder = AutoEncoder().to(device)
classifier = Classifier().to(device)

auto_encoder_criterion = nn.MSELoss()
classifier_criterion = nn.CrossEntropyLoss()

auto_encoder_optimizer = torch.optim.SGD(auto_encoder.parameters(), lr=0.01)
classifier_optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01)

# %%


def train(epochs=20):
    path = "model"
    net_name = f"{auto_encoder._get_name()}{classifier._get_name()}-80"
    criterion_name = f"{auto_encoder_criterion.__class__.__name__}{classifier_criterion.__class__.__name__}"
    optimizer_name = f"{auto_encoder_optimizer.__class__.__name__}{classifier_optimizer.__class__.__name__}"
    model_config_name = f"{net_name}-optimizer-{optimizer_name}-loss-{criterion_name}"
    os.makedirs(path, exist_ok=True)
    state_file_name = f"{path}/state-{model_config_name}.pth"

    state_res = {}
    best_auto_encoder = None
    best_classifier = None

    print(state_file_name, end=" ")
    if os.path.exists(state_file_name):
        print("exist")
        state = torch.load(state_file_name)
        auto_encoder.load_state_dict(state["auto_encoder"])
        classifier.load_state_dict(state["classifier"])
        best_auto_encoder = state.get("best_auto_encoder", None)
        best_classifier = state.get("best_classifier", None)
        auto_encoder_optimizer.load_state_dict(state["auto_encoder_optimizer"])
        classifier_optimizer.load_state_dict(state["classifier_optimizer"])
        # optimizer.load_state_dict(state["optimizer"])
        state_res = state["res"]
    else:
        print("Not exist")
    res = {
        "loss_train": state_res.get("loss_train", []),
        "loss_val": state_res.get("loss_val", []),
        "acc_val": state_res.get("acc_val", []),
        "acc_train": state_res.get("acc_train", []),
        "epoch": state_res.get("epoch", 0),
    }
    best_val_loss = min(res["loss_val"] + [9999])

    a = data_loader["train"]
    num_print = 4
    print_period = len(a) // num_print
    print(f"start epoch {res['epoch']}")
    try:
        # loop over the dataset multiple times
        state_update = False
        for epoch in range(res["epoch"]+1, epochs+1):
            start_time = time.time()
            running_loss = 0.0
            phase_loss = 0
            for phase in ["train", "val"]:

                total, correct = (0, 0)
                if phase == "train":
                    auto_encoder.train(True)  # set model to training mode
                    classifier.train(True)  # Set model to training mode
                else:
                    auto_encoder.train(False)  # set model to training mode
                    classifier.train(False)  # Set model to training mode
                loop_iter = tqdm(enumerate(data_loader[phase], 0), total=len(
                    data_loader[phase]), leave=False)
                for i, data in loop_iter:
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    now_batch_size = inputs.size(0)
                    # zero the parameter gradients
                    auto_encoder_optimizer.zero_grad()
                    classifier_optimizer.zero_grad()
                    # forward + backward + optimize
                    noise = torch.randn(size=inputs.shape, device=device)
                    noisy_data = inputs + noise
                    decoded = auto_encoder(noisy_data)
                    outputs = classifier(decoded)
                    loss1 = auto_encoder_criterion(decoded, inputs)
                    loss2 = classifier_criterion(outputs, labels)
                    loss = loss1 + loss2

                    if phase == "train":
                        loss.backward()
                        auto_encoder_optimizer.step()
                        classifier_optimizer.step()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    # print statistics
                    phase_loss += loss.item() * now_batch_size
                    running_loss += loss.item()
                    loop_iter.set_postfix({"loss": running_loss})
                    # if i % print_period == print_period-1 and phase == "train":
                    #     print(
                    #         f'[{epoch}, {i + 1:5d}] loss: {running_loss / 2000:.4f}')
                    #     running_loss = 0.0
                phase_loss = phase_loss / dataset_sizes[phase]
                if phase == "val" and phase_loss <= best_val_loss:
                    best_val_loss = phase_loss
                    best_auto_encoder = auto_encoder.state_dict()
                    best_classifier = classifier.state_dict()
                    print("### BETTER NET STATE ###")
                # y_loss[phase].append(phase_loss)
                res[f"loss_{phase}"].append(phase_loss)
                res[f"acc_{phase}"].append(round(100*correct/total, 2))
            res["epoch"] = epoch

            end_time = time.time() - start_time
            print(
                f"[{end_time:.0f}s] Epoch {epoch} loss : {res['loss_train'][-1]:.8f} acc: {res['acc_train'][-1]} val: {res['loss_val'][-1]:.8f} acc: {res['acc_val'][-1]}%")
            draw_loss_curve(epoch, name=model_config_name, res=res)
            draw_acc_curve(epoch, name=model_config_name, res=res)

            state_update = False
            state = {
                "epoch": epoch,
                # "state_dict": net.state_dict(),
                "auto_encoder": auto_encoder.state_dict(),
                "classifier": classifier.state_dict(),
                "best_auto_encoder": best_auto_encoder,
                "best_classifier": best_classifier,
                "auto_encoder_optimizer": auto_encoder_optimizer.state_dict(),
                "classifier_optimizer": classifier_optimizer.state_dict(),
                "res": res
            }
            state_update = True

        torch.save(state, state_file_name)
    except KeyboardInterrupt:
        print("Stopping, ", end="")
        if state_update:
            print("Saving")
            torch.save(state, state_file_name)
        else:
            print("Not Saving")
            

# %%


train(151)
# train = TrainModel(
#     nets=[auto_encoder, classifier],
#     optimizers=[auto_encoder_optimizer, classifier_optimizer],
#     criterions=[auto_encoder_criterion, classifier_criterion],
#     train_loader=data_loader["train"],
#     val_loader=data_loader["val"],
#     train_set=dataset["train"],
#     val_set=dataset["val"],
# )
# train.train(301)

# %%
