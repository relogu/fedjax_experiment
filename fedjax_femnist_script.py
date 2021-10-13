import argparse
import pathlib
import json
import itertools
import numpy as np
from typing import Callable, Sequence, Tuple
from timeit import default_timer as timer
import pandas as pd

import fedjax
from fedjax.core import metrics
from fedjax.core import models
import jax
import haiku as hk
import jax.numpy as jnp
import tensorflow as tf

ClientId = bytes
Grads = fedjax.Params

@fedjax.dataclass
class ServerState:
    """State of server passed between rounds.
    Attributes:
    params: A pytree representing the server model parameters.
    opt_state: A pytree representing the server optimizer state.
    """
    params: fedjax.Params
    opt_state: fedjax.OptState


# Adapted to the fedjax federated averaging to perform more than one local epoch
def federated_averaging(
    grad_fn: Callable[[fedjax.Params, fedjax.BatchExample, fedjax.PRNGKey],
                      Grads],
    client_optimizer: fedjax.Optimizer,
    server_optimizer: fedjax.Optimizer,
    client_batch_hparams: fedjax.ShuffleRepeatBatchHParams,
    local_epochs: int = 1
):# -> fedjax.FederatedAlgorithm:
    """Builds the basic implementation of federated averaging."""
    
    def init(params: fedjax.Params):# -> ServerState:
        opt_state = server_optimizer.init(params)
        return ServerState(params, opt_state)
    
    def apply(
        server_state: ServerState,
        clients: Sequence[Tuple[ClientId, fedjax.ClientDataset, fedjax.PRNGKey]]
        ):# -> Tuple[ServerState, Mapping[ClientId, Any]]:
        client_diagnostics = {}
        # We use a list here for clarity, but we strongly recommend avoiding loading
        # all client outputs into memory since the outputs can be quite large
        # depending on the size of the model.
        client_delta_params_weights = []
        for client_id, client_dataset, client_rng in clients:
            delta_params = client_update(server_state.params, client_dataset,
                                         client_rng)
            client_delta_params_weights.append((delta_params, len(client_dataset)))
        # We record the l2 norm of client updates as an example, but it is not
        # required for the algorithm.
        client_diagnostics[client_id] = {
            'delta_l2_norm': fedjax.tree_util.tree_l2_norm(delta_params)
            }
        mean_delta_params = fedjax.tree_util.tree_mean(client_delta_params_weights)
        server_state = server_update(server_state, mean_delta_params)
        return server_state, client_diagnostics

    def client_update(server_params, client_dataset, client_rng):
        params = server_params
        opt_state = client_optimizer.init(params)
        for _ in range(local_epochs):
            for batch in client_dataset.shuffle_repeat_batch(client_batch_hparams):
                client_rng, use_rng = jax.random.split(client_rng)
                grads = grad_fn(params, batch, use_rng)
                opt_state, params = client_optimizer.apply(grads, opt_state, params)
        delta_params = jax.tree_util.tree_multimap(lambda a, b: a - b,
                                                   server_params, params)
        return delta_params

    def server_update(server_state, mean_delta_params):
        opt_state, params = server_optimizer.apply(mean_delta_params,
                                                   server_state.opt_state,
                                                   server_state.params)
        return ServerState(params, opt_state)

    return fedjax.FederatedAlgorithm(init, apply)

class ConvModule(hk.Module):
    """Adapted from haiku module for CNN with dropout to match LEAF model.
    
    This must be defined as a custom hk.Module because only a single positional
    argument is allowed when using hk.Sequential.
    """
    
    def __init__(self, num_classes):
        super().__init__()
        self._num_classes = num_classes
        
    def __call__(self, x: jnp.ndarray):
        x = hk.Reshape(output_shape=(28, 28, 1))(x)
        x = hk.Conv2D(output_channels=32, kernel_shape=(5, 5), padding='SAME')(x)
        x = jax.nn.relu(x)
        x = (
            hk.MaxPool(
                window_shape=(1, 2, 2, 1), # equivalent to pool_size=[2, 2] of tf
                strides=(1, 2, 2, 1), # equivalent to strides=2 of tf
                padding='VALID')(x))
        x = hk.Conv2D(output_channels=64, kernel_shape=(5, 5), padding='SAME')(x)
        x = jax.nn.relu(x)
        x = (
            hk.MaxPool(
                window_shape=(1, 2, 2, 1),
                strides=(1, 2, 2, 1),
                padding='VALID')(x))
        x = hk.Flatten()(x)
        x = hk.Linear(2048)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self._num_classes)(x)
        return x


# Defines the expected structure of input batches to the model. This is used to
# determine the model parameter shapes.
_HAIKU_SAMPLE_BATCH = {
    #'x': np.zeros((1, 28, 28, 1), dtype=np.float32),
    'x': np.zeros((1, 784, 1), dtype=np.float32),
    'y': np.zeros(1, dtype=np.float32)
}
_TRAIN_LOSS = lambda b, p: metrics.unreduced_cross_entropy_loss(b['y'], p)
_EVAL_METRICS = {
    'loss': metrics.CrossEntropyLoss(),
    'accuracy': metrics.Accuracy()
}


def create_conv_model(only_digits: bool = False):# -> models.Model:
    """Creates custom EMNIST CNN model with haiku.
    
    Args:
        only_digits: Whether to use only digit classes [0-9] or include lower and
            upper case characters for a total of 62 classes.
    
    Returns:
        Model
        """
    
    num_classes = 10 if only_digits else 62
    
    def forward_pass(batch):
        return ConvModule(num_classes)(batch['x'])
    
    transformed_forward_pass = hk.transform(forward_pass)
    
    return models.create_model_from_haiku(
        transformed_forward_pass=transformed_forward_pass,
        sample_batch=_HAIKU_SAMPLE_BATCH,
        train_loss=_TRAIN_LOSS,
        eval_metrics=_EVAL_METRICS)


def get_parser():
    parser = argparse.ArgumentParser(
        description="TFF experiment for simulating the FEMNIST training"
    )
    parser.add_argument(
        "--cuda",
        dest="cuda",
        action="store_true",
        help="Flag for hardware acceleration using cuda (if available)",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=2000,
        required=False,
        action="store",
        help="Number of federated rounds to simulate",
    )
    parser.add_argument(
        "--num_clients",
        dest="num_clients",
        type=int,
        default=180,
        required=False,
        action="store",
        help="Number of clients to simulate",
    )
    parser.add_argument(
        "--clients_per_round",
        dest="clients_per_round",
        type=int,
        default=3,
        action="store",
        help="Number of clients per epoch",
    )
    parser.add_argument(
        "--local_epochs",
        dest="local_epochs",
        type=int,
        default=1,
        required=False,
        action="store",
        help="Number of epochs for local clients training",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        required=False,
        action="store_true",
        help="Flag for verbosity",
    )
    return parser


if __name__ == "__main__":
    # Get parameters
    args = get_parser().parse_args()
    # Setting seed for reproducibility
    np.random.seed(51550)

    # This was set to disable GPU when running on my laptop
    if not args.cuda:
        # Disable possible gpu devices for this kernel
        tf.config.set_visible_devices([], "GPU")
        if args.verbose:
            print("GPU disabled")

    # Get arguments
    NUM_CLIENTS = args.num_clients
    NUM_LOCAL_EPOCHS = args.local_epochs
    
    data_path = pathlib.Path('datasets/femnist/preprocessed')
    train_path = data_path/'train'
    test_path = data_path/'test'
    
    # Loading list of paths
    clients_train_path = []
    clients_test_path = []
    for path in train_path.glob('*.json'):
        clients_train_path.append(path)
    for path in test_path.glob('*.json'):
        clients_test_path.append(path)
        
    # Create the model
    model = create_conv_model()
    
    # Set initial model parameters
    LOCAL_EPOCHS = args.local_epochs
    rng = jax.random.PRNGKey(0)
    init_params = model.init(rng)
    # Federated algorithm requires a gradient function, client optimizer, 
    # server optimizers, and hyperparameters for batching at the client level.
    grad_fn = fedjax.model_grad(model)
    client_optimizer = fedjax.optimizers.sgd(0.004)
    server_optimizer = fedjax.optimizers.sgd(1.0)
    batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=20)
    fed_alg = federated_averaging(grad_fn,
                                  client_optimizer,
                                  server_optimizer,
                                  batch_hparams,
                                  LOCAL_EPOCHS)
    
    # Start the timer
    print("Timer has started")
    start_training = timer()
    server_state = fed_alg.init(init_params)
    
    # Select 5 client_ids and their data
    CLIENTS_PER_ROUND = args.clients_per_round
    NUM_ROUNDS = args.num_rounds
    for round_num in range(1, NUM_ROUNDS + 1):
        start_epoch = timer()
        # Get current clients
        idx = np.random.choice(range(len(clients_train_path)), size=CLIENTS_PER_ROUND)
        c_train = [clients_train_path[index] for index in idx]
        c_test = [clients_test_path[index] for index in idx]
        
        # Train set
        client_id_to_train_dataset_mapping = {}
        for i, path in enumerate(c_train):
            with open(path, 'r') as file:
                f = json.load(file)
                client_id_to_train_dataset_mapping[i] = {'x': np.array(f['x']),
                                                         'y': np.array(f['y'])}
        current_train_data = fedjax.InMemoryFederatedData(
            client_id_to_train_dataset_mapping)
        # Test set
        client_id_to_test_dataset_mapping = {}
        for i, path in enumerate(c_test):
            with open(path, 'r') as file:
                f = json.load(file)
                client_id_to_test_dataset_mapping[i] = {'x': np.array(f['x']),
                                                        'y': np.array(f['y'])}
        current_test_data = fedjax.InMemoryFederatedData(
            client_id_to_test_dataset_mapping)
        
        # Prepare clients' datasets
        client_ids = list(current_train_data.client_ids())
        clients_ids_and_data = list(current_train_data.get_clients(client_ids))
        client_inputs = []
        for i in range(CLIENTS_PER_ROUND):
            rng, use_rng = jax.random.split(rng)
            client_id, client_data = clients_ids_and_data[i]
            client_inputs.append((client_id, client_data, use_rng))
            
        # Train clients
        server_state, client_diagnostics = fed_alg.apply(server_state,
                                                         client_inputs)
        
        # # Prints the l2 norm of gradients as part of client_diagnostics. 
        # print(client_diagnostics)
        # Evaluate on "aggregated dataset"
        batched_test_data = list(itertools.islice(
            fedjax.padded_batch_federated_data(current_test_data, batch_size=20),
            CLIENTS_PER_ROUND))
        eval_results = fedjax.evaluate_model(model,
                                             server_state.params,
                                             batched_test_data)
        end_epoch = timer()
        print('Round {:2d}, time elapsed={}'.format(round_num, float(end_epoch - start_epoch)),
              eval_results)


    # End the timer
    end_training = timer()
    print("Timer has stopped")

    # Output results
    res = {
        "framework": "FEDJAX",
        "federated epochs": NUM_LOCAL_EPOCHS,
        "local epochs": NUM_LOCAL_EPOCHS,
        "number of clients": NUM_CLIENTS,
        "number of clients per round": CLIENTS_PER_ROUND,
        "time elapsed": float(end_training - start_training),
        "final loss": eval_results["loss"],
        "final accuracy": eval_results["accuracy"],
    }
    results = pd.DataFrame([res])
    results.to_csv(
        f"logdir/FEDJAX_FEMNIST_{args.clients_per_round}_{args.local_epochs}_results.csv"
    )
