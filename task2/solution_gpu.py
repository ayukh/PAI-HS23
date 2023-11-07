import abc
import collections
import enum
import math
import pathlib
import typing
import warnings

import numpy as np
import torch
import torch.optim
import torch.utils.data
import tqdm
from matplotlib import pyplot as plt

from util import draw_reliability_diagram, cost_function, setup_seeds, calc_calibration_curve

EXTENDED_EVALUATION = False
"""
Set `EXTENDED_EVALUATION` to `True` in order to generate additional plots on validation data.
"""

USE_PRETRAINED_INIT = True
"""
If `USE_PRETRAINED_INIT` is `True`, then MAP inference uses provided pretrained weights.
You should not modify MAP training or the CNN architecture before passing the hard baseline.
If you set the constant to `False` (to further experiment),
this solution always performs MAP inference before running your SWAG implementation.
Note that MAP inference can take a long time.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    """raise RuntimeError(
        "This main() method is for illustrative purposes only"
        " and will NEVER be called when running your solution to generate your submission file!\n"
        "The checker always directly interacts with your SWAGInference class and evaluate method.\n"
        "You can remove this exception for local testing, but be aware that any changes to the main() method"
        " are ignored when generating your submission file."
    )"""

    data_dir = pathlib.Path.cwd()
    model_dir = pathlib.Path.cwd()
    output_dir = pathlib.Path.cwd()

    # Load training data
    train_xs = torch.from_numpy(np.load(data_dir / "train_xs.npz")["train_xs"]).to(device)
    raw_train_meta = np.load(data_dir / "train_ys.npz")
    train_ys = torch.from_numpy(raw_train_meta["train_ys"]).to(device)
    train_is_snow = torch.from_numpy(raw_train_meta["train_is_snow"]).to(device)
    train_is_cloud = torch.from_numpy(raw_train_meta["train_is_cloud"]).to(device)
    dataset_train = torch.utils.data.TensorDataset(train_xs, train_is_snow, train_is_cloud, train_ys)

    # Load validation data
    val_xs = torch.from_numpy(np.load(data_dir / "val_xs.npz")["val_xs"]).to(device)
    raw_val_meta = np.load(data_dir / "val_ys.npz")
    val_ys = torch.from_numpy(raw_val_meta["val_ys"]).to(device)
    val_is_snow = torch.from_numpy(raw_val_meta["val_is_snow"]).to(device)
    val_is_cloud = torch.from_numpy(raw_val_meta["val_is_cloud"]).to(device)
    dataset_val = torch.utils.data.TensorDataset(val_xs, val_is_snow, val_is_cloud, val_ys)

    # Fix all randomness
    setup_seeds()

    # Build and run the actual solution
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    swag = SWAGInference(
        train_xs=dataset_train.tensors[0],
        model_dir=model_dir,
        swag_epochs=300,
        bma_samples=50
    )

    swag.fit(train_loader)
    swag.calibrate(dataset_val)

    # fork_rng ensures that the evaluation does not change the rng state.
    # That way, you should get exactly the same results even if you remove evaluation
    # to save computational time when developing the task
    # (as long as you ONLY use torch randomness, and not e.g. random or numpy.random).
    
    with torch.random.fork_rng():
        evaluate(swag, dataset_val, EXTENDED_EVALUATION, output_dir)


class InferenceMode(enum.Enum):
    """
    Inference mode switch for your implementation.
    `MAP` simply predicts the most likely class using pretrained MAP weights.
    `SWAG_DIAGONAL` and `SWAG_FULL` correspond to SWAG-diagonal and the full SWAG method, respectively.
    """
    MAP = 0
    SWAG_DIAGONAL = 1
    SWAG_FULL = 2


class SWAGInference(object):
    """
    Your implementation of SWA-Gaussian.
    This class is used to run and evaluate your solution.
    You must preserve all methods and signatures of this class.
    However, you can add new methods if you want.

    We provide basic functionality and some helper methods.
    You can pass all baselines by only modifying methods marked with TODO.
    However, we encourage you to skim other methods in order to gain a better understanding of SWAG.
    """

    def __init__(
        self,
        train_xs: torch.Tensor,
        model_dir: pathlib.Path,
        # TODO(1): change inference_mode to InferenceMode.SWAG_DIAGONAL
        # TODO(2): change inference_mode to InferenceMode.SWAG_DIAGONAL,
        inference_mode: InferenceMode = InferenceMode.SWAG_DIAGONAL,
        # TODO(2): change inference_mode to InferenceMode.SWAG_FULL
        # TODO(2): change inference_mode to InferenceMode.SWAG_FULL
        # TODO(2): change inference_mode to InferenceMode.SWAG_FULL
        inference_mode: InferenceMode = InferenceMode.SWAG_FULL,
        # TODO(2): optionally add/tweak hyperparameters
        swag_epochs: int = 300,
        swag_learning_rate: float = 1e-3#0.045,
        swag_update_freq: int = 1,
        deviation_matrix_max_rank: int = 15,
        bma_samples: int = 30,
    ):
        """
        :param train_xs: Training images (for storage only)
        :param model_dir: Path to directory containing pretrained MAP weights
        :param inference_mode: Control which inference mode (MAP, SWAG-diagonal, full SWAG) to use
        :param swag_epochs: Total number of gradient descent epochs for SWAG
        :param swag_learning_rate: Learning rate for SWAG gradient descent
        :param swag_update_freq: Frequency (in epochs) for updating SWAG statistics during gradient descent
        :param deviation_matrix_max_rank: Rank of deviation matrix for full SWAG
        :param bma_samples: Number of networks to sample for Bayesian model averaging during prediction
        """

        self.model_dir = model_dir
        self.inference_mode = inference_mode
        self.swag_epochs = swag_epochs
        self.swag_learning_rate = swag_learning_rate
        self.swag_update_freq = swag_update_freq
        self.deviation_matrix_max_rank = deviation_matrix_max_rank
        self.bma_samples = bma_samples

        # Network used to perform SWAG.
        # Note that all operations in this class modify this network IN-PLACE!
        self.network = CNN(in_channels=3, out_classes=6).to(device)

        # Store training dataset to recalculate batch normalization statistics during SWAG inference
        self.train_dataset = torch.utils.data.TensorDataset(train_xs)

        # SWAG-diagonal
        # TODO(1): create attributes for SWAG-diagonal
        #  Hint: self._create_weight_copy() creates an all-zero copy of the weights
        #  as a dictionary that maps from weight name to values.
        #  Hint: you never need to consider the full vector of weights,
        #  but can always act on per-layer weights (in the format that _create_weight_copy() returns)
        
        self.diagonal_swag = []

        # Full SWAG
        # TODO(2): create attributes for SWAG-diagonal
        #  Hint: check collections.deque

        # Calibration, prediction, and other attributes
        # TODO(2): create additional attributes, e.g., for calibration
        self._prediction_threshold = None  # this is an example, feel free to be creative

    def update_swag(self) -> None:
        """
        Update SWAG statistics with the current weights of self.network.
        """

        # Create a copy of the current network weights
        current_params = {name: param.detach() for name, param in self.network.named_parameters()}

        # SWAG-diagonal
        for name, param in current_params.items():
            # TODO(1): update SWAG-diagonal attributes for weight `name` using `current_params` and `param`
            wc = self._create_weight_copy()
            self.diagonal_swag.append(wc)
            # update diag swag for weight "name"
            self.diagonal_swag[-1][name] = param

        # Full SWAG
        if self.inference_mode == InferenceMode.SWAG_FULL:
            # TODO(2): update full SWAG attributes for weight `name` using `current_params` and `param`
            raise NotImplementedError("Update full SWAG statistics")

    def fit_swag(self, loader: torch.utils.data.DataLoader) -> None:
        """
        Fit SWAG on top of the pretrained network self.network.
        This method should perform gradient descent with occasional SWAG updates
        by calling self.update_swag().
        """

        # We use SGD with momentum and weight decay to perform SWA.
        # See the paper on how weight decay corresponds to a type of prior.
        # Feel free to play around with optimization hyperparameters.
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.swag_learning_rate,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-5, #1e-4
        )
        loss = torch.nn.CrossEntropyLoss(
            reduction="mean",
        )
        # TODO(2): Update SWAGScheduler instantiation if you decided to implement a custom schedule.
        #  By default, this scheduler just keeps the initial learning rate given to `optimizer`.
        lr_scheduler = SWAGScheduler(
            optimizer,
            epochs=self.swag_epochs,
            steps_per_epoch=len(loader),
        )

        # TODO(1): Perform initialization for SWAG fitting

        self.network.train()
        with tqdm.trange(self.swag_epochs, desc="Running gradient descent for SWA") as pbar:
            pbar_dict = {}
            for epoch in pbar:
                average_loss = 0.0
                average_accuracy = 0.0
                num_samples_processed = 0
                for batch_xs, batch_is_snow, batch_is_cloud, batch_ys in loader:
                    optimizer.zero_grad()
                    pred_ys = self.network(batch_xs)
                    batch_loss = loss(input=pred_ys, target=batch_ys)
                    batch_loss.backward()
                    optimizer.step()
                    pbar_dict["lr"] = lr_scheduler.get_last_lr()[0]
                    lr_scheduler.step()

                    # Calculate cumulative average training loss and accuracy
                    average_loss = (batch_xs.size(0) * batch_loss.item() + num_samples_processed * average_loss) / (
                        num_samples_processed + batch_xs.size(0))
                    average_accuracy = (
                        torch.sum(pred_ys.argmax(dim=-1) == batch_ys).item()
                        + num_samples_processed * average_accuracy
                    ) / (num_samples_processed + batch_xs.size(0))
                    num_samples_processed += batch_xs.size(0)
                    pbar_dict["avg. epoch loss"] = average_loss
                    pbar_dict["avg. epoch accuracy"] = average_accuracy
                    pbar.set_postfix(pbar_dict)

                # TODO(1): Implement periodic SWAG updates using the attributes defined in __init__
                self.update_swag()

    def calibrate(self, validation_data: torch.utils.data.Dataset) -> None:
        """
        Calibrate your predictions using a small validation set.
        validation_data contains well-defined and ambiguous samples,
        where you can identify the latter by having label -1.
        """
        if self.inference_mode == InferenceMode.MAP:
            # In MAP mode, simply predict argmax and do nothing else
            self._prediction_threshold = 0.0
            return

        # TODO(1): pick a prediction threshold, either constant or adaptive.
        #  The provided value should suffice to pass the easy baseline.
        self._prediction_threshold = 2.0 / 3.0

        # TODO(2): perform additional calibration if desired.
        #  Feel free to remove or change the prediction threshold.
        val_xs, val_is_snow, val_is_cloud, val_ys = validation_data.tensors
        assert val_xs.size() == (140, 3, 60, 60)  # N x C x H x W
        assert val_ys.size() == (140,)
        assert val_is_snow.size() == (140,)
        assert val_is_cloud.size() == (140,)

        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        self.network.eval()
        sigmoid = torch.nn.Sigmoid()
        with torch.no_grad():
            pred_probs = []
            for batch in tqdm.tqdm(val_loader, desc="Calibrating predictions"):
                batch_xs, _, _, _ = batch
                logits = self.network(batch_xs)
                probs = sigmoid(logits)
                pred_probs.append(probs)
        
        pred_probs = torch.cat(pred_probs)
        true_probs = pred_probs[np.arange(len(pred_probs)), val_ys]
        # Calculate ECE and plot the calibration curve
        ece = calc_ece(pred_probs.cpu().numpy(), val_ys.cpu().numpy(), num_bins=10)
        print("Validation ECE:", ece)

        if extended_evaluation:
            print("Plotting reliability diagram")
            fig = draw_reliability_diagram(pred_probs.cpu().numpy(), val_ys.cpu().numpy(), num_bins=10)
            fig.savefig(output_dir / "reliability_diagram.pdf")

            sorted_confidence_indices = torch.argsort(pred_probs.max(dim=1).values)

            # Plot samples your model is most confident about
            print("Plotting most confident validation set predictions")
            most_confident_indices = sorted_confidence_indices[-10:]
            fig, ax = plt.subplots(4, 5, figsize=(13, 11))
            for row in range(0, 4, 2):
                for col in range(5):
                    sample_idx = most_confident_indices[5 * row // 2 + col]
                    ax[row, col].imshow(val_xs[sample_idx].permute(1, 2, 0).cpu().numpy())
                    ax[row, col].set_axis_off()
                    ax[row + 1, col].set_title(f"pred. {pred_ys[sample_idx]}, true {val_ys[sample_idx]}")
                    bar_colors = ["C0"] * 6
                    if val_ys[sample_idx] >= 0:
                        bar_colors[val_ys[sample_idx]] = "C1"
                    ax[row + 1, col].bar(
                        np.arange(6), pred_probs[sample_idx].cpu().numpy(), tick_label=np.arange(6), color=bar_colors
                    )
            fig.suptitle("Most confident predictions", size=20)
            fig.savefig(output_dir / "examples_most_confident.pdf")

            # Plot samples your model is least confident about
            print("Plotting least confident validation set predictions")
            least_confident_indices = sorted_confidence_indices[:10]
            fig, ax = plt.subplots(4, 5, figsize=(13, 11))
            for row in range(0, 4, 2):
                for col in range(5):
                    sample_idx = least_confident_indices[5 * row // 2 + col]
                    ax[row, col].imshow(val_xs[sample_idx].permute(1, 2, 0).cpu().numpy())
                    ax[row, col].set_axis_off()
                    ax[row + 1, col].set_title(f"pred. {pred_ys[sample_idx]}, true {val_ys[sample_idx]}")
                    bar_colors = ["C0"] * 6
                    if val_ys[sample_idx] >= 0:
                        bar_colors[val_ys[sample_idx]] = "C1"
                    ax[row + 1, col].bar(
                        np.arange(6), pred_probs[sample_idx].cpu().numpy(), tick_label=np.arange(6), color=bar_colors
                    )
            fig.suptitle("Least confident predictions", size=20)
            fig.savefig(output_dir / "examples_least_confident.pdf")


class CNN(torch.nn.Module):
    """
    Small convolutional neural network used in this task.
    You should not modify this class before passing the hard baseline.

    Note that if you change the architecture of this network,
    you need to re-run MAP inference and cannot use the provided pretrained weights anymore.
    Hence, you need to set `USE_PRETRAINED_INIT = False` at the top of this file.
    """

    def __init__(
        self,
        in_channels: int,
        out_classes: int,
    ) -> None:
        super().__init__()

        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=5),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.pool1 = torch.nn.MaxPool2d((2, 2), stride=(2, 2))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.pool2 = torch.nn.MaxPool2d((2, 2), stride=(2, 2))

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
        )

        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.linear = torch.nn.Linear(64, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool1(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool2(x)
        x = self.layer5(x)

        # Average features over both spatial dimensions, and remove the now superfluous dimensions
        x = self.global_pool(x).squeeze(-1).squeeze(-1)

        # Note: this network does NOT output the per-class probabilities y =[y_1, ..., y_C],
        # but a feature vector z such that y = softmax(z).
        # This avoids numerical instabilities during optimization.
        # The PyTorch loss automatically handles this.
        log_softmax = self.linear(x)

        return log_softmax
    
if __name__ == "__main__":
    main()