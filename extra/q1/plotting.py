# %%
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import pathlib
import seaborn as sns
from sklearn.metrics import classification_report

# %%


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


def plot_conf_matrix(y_true, y_pred, name):
    plt.clf()
    cm = confusion_matrix(y_true, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[0, 1])
    cm_display.plot()
    os.makedirs("loss_graphs", exist_ok=True)
    plt.savefig(f"./loss_graphs/cm-{name}.png")
    plt.clf()


def plot_classification_report(y_test, y_pred, title='Classification Report', figsize=(8, 6), dpi=70, name=None, **kwargs):
    save_fig_path = f"loss_graphs/classification_report_{name}.png"
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    clf_report = classification_report(
        y_test, y_pred, output_dict=True, **kwargs)
    keys_to_plot = [key for key in clf_report.keys() if key not in (
        'accuracy', 'macro avg', 'weighted avg')]
    df = pd.DataFrame(clf_report, columns=keys_to_plot).T
    df.sort_values(by=['support'], inplace=True)

    rows, cols = df.shape
    mask = np.zeros(df.shape)
    mask[:, cols-1] = True

    ax = sns.heatmap(df, mask=mask, annot=True, cmap="YlGn", fmt='.3g',
                     vmin=0.0,
                     vmax=1.0,
                     linewidths=2, linecolor='white'
                     )

    # then, let's add the support column by normalizing the colors in this column
    mask = np.zeros(df.shape)
    mask[:, :cols-1] = True

    ax = sns.heatmap(df, mask=mask, annot=True, cmap="YlGn", cbar=False,
                     linewidths=2, linecolor='white', fmt='.0f',
                     vmin=df['support'].min(),
                     vmax=df['support'].sum(),
                     norm=mpl.colors.Normalize(vmin=df['support'].min(),
                                               vmax=df['support'].sum())
                     )

    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=360)

    if (save_fig_path != None):
        path = pathlib.Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path)

    return fig, ax
