import os
from typing import List
import time
from tqdm import tqdm
import torch
import torch.nn as nn

import torch.optim.optimizer as optimizer
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


class TrainModel():
    phases = ["train", "val"]

    def __init__(
            self,
            nets: List[nn.Module],
            criterions: List[nn.modules.loss._Loss],
            optimizers: List[optimizer.Optimizer],
            train_loader: DataLoader,
            val_loader: DataLoader,
            train_set: Dataset,
            val_set: Dataset,
    ):
        if not isinstance(nets, list):
            self.nets = [nets]
        else:
            self.nets = nets
        if not isinstance(criterions, list):
            self.criterions = [criterions]
        else:
            self.criterions = criterions
        if not isinstance(optimizers, list):
            self.optimizers = [optimizers]
        else:
            self.optimizers = optimizers

        self.data_loader = {
            "train": train_loader,
            "val": val_loader,
        }
        self.dataset_sizes = {
            "train": len(train_set),
            "val": len(val_set),
        }
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.first_loss_plot = True
        self.first_acc_plot = True
        self.base_path = "models"
        os.makedirs(self.base_path, exist_ok=True)

    def get_obj_name(self, obj) -> str:
        if isinstance(obj, nn.Module):
            return obj._get_name()
        elif isinstance(obj, optimizer.Optimizer):
            return obj.__class__.__name__

    def get_name(self, objs) -> str:
        name = ""
        for obj in objs:
            name += f"{self.get_obj_name(obj)}"
        return name

    @property
    def net_name(self) -> str:
        return self.get_name(self.nets)

    @property
    def criterion_name(self) -> str:
        return self.get_name(self.criterions)

    @property
    def optimizer_name(self) -> str:
        return self.get_name(self.optimizers)

    @property
    def state_file(self) -> str:
        name = f"{self.base_path}/state-{self.net_name}-optimizer-{self.optimizer_name}-loss-{self.criterion_name}.pth"
        return name

    def draw_loss_curve(self, current_epoch, res):
        global first_loss
        plt.clf()
        x_epoch = list(range(1, current_epoch+1))
        loss_train = res["loss_train"]
        loss_val = res["loss_val"]
        plt.plot(x_epoch, loss_train, 'bo-', label='train')
        plt.plot(x_epoch, loss_val, 'ro-', label='val')
        if self.first_loss_plot:
            plt.legend()
            self.first_loss_plot = False
        os.makedirs("loss_graphs", exist_ok=True)
        plt.savefig(os.path.join('./loss_graphs',
                    f'train_loss_{self.net_name}-{self.optimizer_name}_{self.criterion_name}.png'))
        plt.clf()

    def draw_acc_curve(self, current_epoch, res):
        plt.clf()
        x_epoch = list(range(1, current_epoch+1))
        acc_train = res["acc_train"]
        acc_val = res["acc_val"]
        plt.plot(x_epoch, acc_train, 'bo-', label='train')
        plt.plot(x_epoch, acc_val, 'ro-', label='val')
        if self.first_acc_plot:
            plt.legend()
            self.first_acc_plot = False
        os.makedirs("loss_graphs", exist_ok=True)
        plt.savefig(os.path.join('./loss_graphs',
                    f'train_acc_{self.net_name}-{self.optimizer_name}_{self.criterion_name}.png'))
        plt.clf()

    def net_train_mode(self, mode):
        for net in self.nets:
            net.train(mode)

    def load_net_states(self, state):
        for net in self.nets:
            net.load_state_dict(state[f"state_dict_{self.get_obj_name(net)}"])

    def save_net_states(self):
        states = {}
        for net in self.nets:
            states[f"state_dict_{self.get_obj_name(net)}"] = net.state_dict()
        return states

    def load_optimizer_states(self, state):
        for optim in self.optimizers:
            optim.load_state_dict(
                state[f"optimizer_{self.get_obj_name(optim)}"])

    def get_optimizers_states(self) -> dict:
        states = {}
        for optim in self.optimizers:
            states[f"optimizer_{self.get_obj_name(optim)}"] = optim.state_dict(
            )
        return states

    def best_net_state(self, state) -> dict:
        return state["best_state_dict"]

    def get_nets_state_dict(self):
        states = {}
        for n in self.nets:
            states[f"state_dict_{self.get_obj_name(n)}"] = n.state_dict()
        return states

    def zero_grad(self):
        for optim in self.optimizers:
            optim.zero_grad()

    def calc_outputs(self, inputs):
        inputs_list = [inputs]
        inputs_dict = {}
        outputs_dict = {}
        last_output = None
        for i, net in enumerate(self.nets, 0):
            net_input = inputs_list[i]
            output = net(net_input)
            inputs_list.append(output)
            outputs_dict[self.get_obj_name(net)] = output
            inputs_dict[self.get_obj_name(net)] = net_input
            last_output = output
        return inputs_dict, outputs_dict, last_output

    def calc_loss(self, loss_inputs_dict):
        losses = []
        for net, criterion in zip(self.nets, self.criterions):
            data1, data2 = loss_inputs_dict[self.get_obj_name(net)]
            losses.append(criterion(data1, data2))
        loss = sum(losses)
        return loss

    def optimizers_step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def train(self, epochs=20):
        state_res = {}
        best_net_state = None

        if os.path.exists(self.state_file):
            print(self.state_file, "exist")
            self.state = torch.load(self.state_file)
            self.load_net_states(self.state)
            self.load_optimizer_states(self.state)
            best_net_state = self.best_net_state(self.state)
            state_res = self.state["res"]

        else:
            print(self.state_file, "not exist")

        res = {
            "loss_train": state_res.get("loss_train", []),
            "loss_val": state_res.get("loss_val", []),
            "acc_val": state_res.get("acc_val", []),
            "acc_train": state_res.get("acc_train", []),
            "epoch": state_res.get("epoch", 0),
        }
        best_val_loss = min(res["loss_val"] + [9999])

        # loop over the dataset multiple times
        try:
            for epoch in range(res["epoch"]+1, epochs+1):
                start_time = time.time()
                running_loss = 0.0
                phase_loss = 0
                for phase in self.phases:
                    total, correct = (0, 0)
                    if phase == "train":
                        # Set model to training mode
                        self.net_train_mode(True)
                    else:
                        # Set model to evaluate mode
                        self.net_train_mode(False)
                    loop_iter = tqdm(enumerate(self.data_loader[phase], 0), total=len(
                        self.data_loader[phase]), leave=False)
                    for i, data in loop_iter:
                        # get the inputs; data is a list of [inputs, labels]
                        inputs, labels = data
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        now_batch_size = inputs.size(0)
                        # zero the parameter gradients
                        self.zero_grad()
                        # forward + backward + optimize
                        inputs_dict, outputs_dict, last_output = self.calc_outputs(inputs)
                        loss_input = {
                            self.get_obj_name(self.nets[0]): [
                                outputs_dict[self.get_obj_name(self.nets[0])], inputs
                            ],
                            self.get_obj_name(self.nets[1]): [
                                outputs_dict[self.get_obj_name(self.nets[1])], labels
                            ]
                        }
                        loss = self.calc_loss(loss_input)
                        if phase == "train":
                            loss.backward()
                            self.optimizers_step()
                        _, predicted = torch.max(last_output.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        # print statistics
                        phase_loss += loss.item() * now_batch_size
                        running_loss += loss.item()
                        loop_iter.set_postfix(
                            {"loss": f"{running_loss / (i+1):04.4f}"})
                    phase_loss = phase_loss / self.dataset_sizes[phase]
                    if phase == "val" and phase_loss <= best_val_loss:
                        best_val_loss = phase_loss
                        best_net_state = self.get_nets_state_dict()
                        print("### BETTER NET STATE ###")
                    # y_loss[phase].append(phase_loss)

                    res[f"acc_{phase}"].append(round(100*correct/total, 2))
                    res[f"loss_{phase}"].append(phase_loss)
                res["epoch"] = epoch
                end_time = time.time() - start_time
                print(
                    f"[{end_time:.0f}s] Epoch {epoch} loss : {res['loss_train'][-1]:.8f} acc: {res['acc_train'][-1]} val: {res['loss_val'][-1]:.8f} acc: {res['acc_val'][-1]}%")
                # if epoch % 5 == 0 or epoch in (1, epochs-1):
                #     total_accuracy()
                self.draw_loss_curve(epoch, res=res)
                self.draw_acc_curve(epoch, res=res)
                self.state = {
                    "epoch": epoch,
                    **self.get_nets_state_dict(),
                    **self.get_optimizers_states(),
                    "best_net_state": best_net_state,
                    "res": res
                }
        except KeyboardInterrupt:
            print("Stopping")
            torch.save(self.state, self.state_file)

        torch.save(self.state, self.state_file)
