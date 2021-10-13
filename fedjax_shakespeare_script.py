import argparse
import pathlib
import json
import itertools
import numpy as np
from typing import Callable, Sequence, Tuple
from timeit import default_timer as timer
import pandas as pd
import os

# Lo
os.environ["XLA_FLAGS"] = '--xla_dump_to=/tmp/foo'
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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


def create_rnn_model(vocab_size: int = 80,
                     embed_size: int = 8,
                     lstm_hidden_size: int = 256,
                     lstm_num_layers: int = 2):# -> models.Model:
    """Creates LSTM RNN language model.

    Character-level LSTM for Shakespeare language model.
    Defaults to the model used in:

    Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629

    Args:
        vocab_size: The number of possible output characters. This does not include
          special tokens like PAD, BOS, EOS, or OOV.
        embed_size: Embedding size for each character.
        lstm_hidden_size: Hidden size for LSTM cells.
        lstm_num_layers: Number of LSTM layers.

    Returns:
        Model.
        """
    # TODO(jaero): Replace these with direct references from dataset.
    pad = 0
    bos = vocab_size + 1
    eos = vocab_size + 2
    oov = vocab_size + 3
    full_vocab_size = vocab_size + 4
    # We do not guess EOS, and if we guess OOV, it's treated as a mistake.
    logits_mask = [0. for _ in range(full_vocab_size)]
    for i in (pad, bos, eos, oov):
        logits_mask[i] = jnp.NINF
    logits_mask = tuple(logits_mask)

    def forward_pass(batch):
        x = batch['x']
        # [time_steps, batch_size, ...].
        x = jnp.transpose(x)
        # [time_steps, batch_size, embed_dim].
        embedding_layer = hk.Embed(full_vocab_size, embed_size)
        embeddings = embedding_layer(x)

        lstm_layers = []
        for _ in range(lstm_num_layers):
          lstm_layers.extend([hk.LSTM(hidden_size=lstm_hidden_size), jnp.tanh])
        rnn_core = hk.DeepRNN(lstm_layers)
        initial_state = rnn_core.initial_state(batch_size=embeddings.shape[1])
        # [time_steps, batch_size, hidden_size].
        output, _ = hk.dynamic_unroll(rnn_core, embeddings, initial_state)

        output = hk.Linear(full_vocab_size)(output)
        # [batch_size, time_steps, full_vocab_size].
        output = jnp.transpose(output, axes=(1, 0, 2))
        return output

    def train_loss(batch, preds):
        """Returns average token loss per sequence."""
        targets = batch['y']
        per_token_loss = metrics.unreduced_cross_entropy_loss(targets, preds)
        # Don't count padded values in loss.
        per_token_loss *= targets != pad
        return jnp.mean(per_token_loss, axis=-1)

    transformed_forward_pass = hk.transform(forward_pass)
    return models.create_model_from_haiku(
        transformed_forward_pass=transformed_forward_pass,
        sample_batch={
            'x': jnp.zeros((1, 80), dtype=jnp.int32),
            'y': jnp.zeros((1, 80), dtype=jnp.int32),
            },
        train_loss=train_loss,
        eval_metrics={
            'accuracy_in_vocab':
                metrics.SequenceTokenAccuracy(
                    masked_target_values=(pad, eos), logits_mask=logits_mask),
            'accuracy_no_eos':
                metrics.SequenceTokenAccuracy(masked_target_values=(pad, eos)),
            'num_tokens':
                metrics.SequenceTokenCount(masked_target_values=(pad,)),
            'sequence_length':
                metrics.SequenceLength(masked_target_values=(pad,)),
            'sequence_loss':
                metrics.SequenceCrossEntropyLoss(masked_target_values=(pad,)),
            'token_loss':
                metrics.SequenceTokenCrossEntropyLoss(
                    masked_target_values=(pad,)),
            'token_oov_rate':
                metrics.SequenceTokenOOVRate(
                    oov_target_values=(oov,), masked_target_values=(pad,)),
                })

# Taken from LEAF code
ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"

# Taken from LEAF code
def word_to_indices(word):
    '''returns a list of character indices

    Args:
        word: string
    
    Return:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices

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
        "--n_rounds",
        type=int,
        default=2000,
        required=False,
        action="store",
        help="Number of federated rounds to simulate",
    )
    parser.add_argument(
        "--clients_per_round",
        dest="clients_per_round",
        type=int,
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
        '-l', '--learning_rate',
        dest='learning_rate',
        type=float,
        default=0.8, # from LEAF paper
        required=False,
        action='store',
        help='Learning rate for SGD in clients training')
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
    NUM_LOCAL_EPOCHS = args.local_epochs
    
    data_path = pathlib.Path('datasets/shakespeare/preprocessed')
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
    model = create_rnn_model()
    
    # Set initial model parameters
    LOCAL_EPOCHS = args.local_epochs
    rng = jax.random.PRNGKey(0)
    init_params = model.init(rng)
    # Federated algorithm requires a gradient function, client optimizer, 
    # server optimizers, and hyperparameters for batching at the client level.
    grad_fn = fedjax.model_grad(model)
    client_optimizer = fedjax.optimizers.sgd(args.learning_rate)
    server_optimizer = fedjax.optimizers.sgd(1.0)
    batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=10)
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
    NUM_ROUNDS = args.n_rounds
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
                x = np.array(f['x'])
                y = np.array(f['y'])
                a = np.array([xx+yy for (xx, yy) in zip(x, y)])
                x = np.array([aa[:-1] for aa in a])
                y = np.array([aa[1:] for aa in a])
                x = np.array([word_to_indices(word) for word in x])
                y = np.array([word_to_indices(word) for word in y])
                client_id_to_train_dataset_mapping[i] = {'x': x,
                                                         'y': y}
        current_train_data = fedjax.InMemoryFederatedData(
            client_id_to_train_dataset_mapping)
        # Test set
        client_id_to_test_dataset_mapping = {}
        for i, path in enumerate(c_test):
            with open(path, 'r') as file:
                f = json.load(file)
                x = np.array(f['x'])
                y = np.array(f['y'])
                a = [xx+yy for (xx, yy) in zip(x, y)]
                x = [aa[:-1] for aa in a]
                y = [aa[1:] for aa in a]
                x = np.array([word_to_indices(word) for word in x])
                y = np.array([word_to_indices(word) for word in y])
                client_id_to_test_dataset_mapping[i] = {'x': x,
                                                        'y': y}
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
        if args.verbose:
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
        "number of clients": len(clients_test_path),
        "number of clients per round": CLIENTS_PER_ROUND,
        "time elapsed": float(end_training - start_training),
        "final loss": eval_results["token_loss"],
        "final accuracy": eval_results["accuracy_in_vocab"],
    }
    results = pd.DataFrame([res])
    results.to_csv(
        f"logdir/FEDJAX_Shakespeare_{args.clients_per_round}_{args.local_epochs}_results.csv"
    )

