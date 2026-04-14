import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def create_tensorboard_writer(
    base_dir: str = "runs",
    run_name: str | None = None,
):
    """
    Create a TensorBoard SummaryWriter with a unique run directory.

    Returns:
        writer (SummaryWriter)
        log_dir (str)
    """
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = os.path.join(base_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    return writer, log_dir