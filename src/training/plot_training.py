from IPython.display import clear_output
import matplotlib.pyplot as plt


def plot_losses(train_losses, test_losses, test_accs):
    clear_output(wait=True)

    plt.plot(train_losses)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(train_losses, label="train loss")
    axes[0].plot(test_losses, label="test loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(test_accs, label="test accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()