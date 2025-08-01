import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
import math
import gc
import os
import golois 
from tensorflow.keras.callbacks import CSVLogger
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ------------------------------------------------------------------
# DataIterator class for streaming training data.
class DataIterator:
    def __init__(self, epochs, N):
        """Initialize iterator with number of epochs per iteration and batch size N."""
        self.epochs = epochs
        self.N = N
        self.index = -1

    def _store_data_in_ram(self):
        """Load a chunk of training data (epochs * N samples) into memory via golois."""
        START = self.index * self.epochs + 1
        END = (self.index + 1) * self.epochs + 1
        print("START", START, "END", END)
        planes = 31
        moves = 361

        # Allocate arrays for one batch of N samples
        input_data = np.random.randint(2, size=(self.N, 19, 19, planes)).astype('float32')
        policy = np.random.randint(moves, size=(self.N,))
        policy = keras.utils.to_categorical(policy, num_classes=moves)
        value = np.random.randint(2, size=(self.N,)).astype('float32')
        end = np.random.randint(2, size=(self.N, 19, 19, 2)).astype('float32')
        groups = np.zeros((self.N, 19, 19, 1)).astype('float32')

        # Load data for each epoch and store in lists      
        inputs_data_list = []
        policy_list = []
        value_list = []
        end_list = []
        groups_list = []

        for i in range(START, END):
            # Fetch a batch of training data via golois (fills the arrays in-place)
            golois.getBatch(input_data, policy, value, end, groups, i * self.N)
            # Store copies of the data for this epoch
            inputs_data_list.append(np.copy(input_data))
            policy_list.append(np.copy(policy))
            value_list.append(np.copy(value))
            end_list.append(np.copy(end))
            groups_list.append(np.copy(groups))

        # Concatenate data from all epochs into one large array
        inputs_data_list = np.concatenate(inputs_data_list, axis=0)
        policy_list = np.concatenate(policy_list, axis=0)
        value_list = np.concatenate(value_list, axis=0)
        end_list = np.concatenate(end_list, axis=0)
        groups_list = np.concatenate(groups_list, axis=0)
        return inputs_data_list, policy_list, value_list, end_list, groups_list

    def __iter__(self):
        return self

    def __next__(self):
        """Load the next training iteration data."""
        self.index += 1
        print("Iteration index:", self.index)
        return self._store_data_in_ram()

    def __getitem__(self, index):
        """Load a specific iteration of training data."""
        self.index = index
        return self._store_data_in_ram()

# ------------------------------------------------------------------
# Define a custom residual block with parallel depthwise convs and SE unit.
def residual_block(x, block_id, se_ratio=0.125):
    """BalancedGoNet residual block: expand -> depthwise convs -> concat -> compress -> SE -> add -> ReLU."""
    shortcut = x  # Save input for residual connection

    # Expansion: 1×1 conv from 32 to 96 channels
    x = layers.Conv2D(96, kernel_size=1, padding='same', use_bias=False, name=f'block{block_id}_conv_expand')(x)
    x = layers.BatchNormalization(name=f'block{block_id}_bn_expand')(x)
    x = layers.Activation('relu', name=f'block{block_id}_act_expand')(x)

    # Parallel depthwise convolutions: 3×3 and 5×5 (no bias)
    d1 = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False, name=f'block{block_id}_dw3')(x)
    d2 = layers.DepthwiseConv2D(kernel_size=5, padding='same', use_bias=False, name=f'block{block_id}_dw5')(x)
    x = layers.Concatenate(name=f'block{block_id}_concat')([d1, d2])
    x = layers.BatchNormalization(name=f'block{block_id}_bn_concat')(x)
    x = layers.Activation('relu', name=f'block{block_id}_act_concat')(x)

    # Compression: 1×1 conv to reduce channels back to 32
    x = layers.Conv2D(32, kernel_size=1, padding='same', use_bias=False, name=f'block{block_id}_conv_compress')(x)
    x = layers.BatchNormalization(name=f'block{block_id}_bn_compress')(x)

    # Squeeze-and-Excitation (SE) block
    se = layers.GlobalAveragePooling2D(name=f'block{block_id}_se_gap')(x)
    se = layers.Dense(max(1, int(32 * se_ratio)), activation='relu', name=f'block{block_id}_se_dense1')(se)
    se = layers.Dense(32, activation='sigmoid', name=f'block{block_id}_se_dense2')(se)
    se = layers.Reshape((1, 1, 32), name=f'block{block_id}_se_reshape')(se)
    x = layers.Multiply(name=f'block{block_id}_se_scale')([x, se])

    # Residual addition and output activation
    x = layers.Add(name=f'block{block_id}_add')([x, shortcut])
    x = layers.Activation('relu', name=f'block{block_id}_act_out')(x)
    return x

# ------------------------------------------------------------------
# BalancedGoNet model architecture.
def create_balanced_model(input_shape=(19, 19, 31), num_blocks=7, num_moves=361):
    """Construct the BalancedGoNet model with the given input shape and number of residual blocks."""
    board_input = keras.Input(shape=input_shape, name='board')

    # Initial convolution: 1×1 conv to 32 channels (with bias)
    x = layers.Conv2D(32, kernel_size=1, padding='same', use_bias=True, name='conv_initial')(board_input)
    x = layers.BatchNormalization(name='bn_initial')(x)
    x = layers.Activation('relu', name='act_initial')(x)

    # Residual tower: apply a series of residual blocks
    for i in range(num_blocks):
        x = residual_block(x, block_id=i+1)

    # Policy head: 1×1 conv to 1 channel, flatten, then softmax over 361 moves
    policy_conv = layers.Conv2D(1, kernel_size=1, padding='same', use_bias=True, name='conv_policy')(x)
    flat_policy = layers.Flatten(name='flatten_policy')(policy_conv)
    policy_output = layers.Activation('softmax', name='policy')(flat_policy)

    # Value head: Global average pool, then two dense layers (50 -> 1) with ReLU and sigmoid
    gap = layers.GlobalAveragePooling2D(name='gap_value')(x)
    dense_val = layers.Dense(50, activation='relu', name='dense_value1')(gap)
    value_output = layers.Dense(1, activation='sigmoid', name='value')(dense_val)

    # Create the model with two outputs
    model = keras.Model(inputs=board_input, outputs=[policy_output, value_output], name='BalancedGoNet')

    # Compile with Adam optimizer (for testing) with enhanced metrics.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
        loss_weights={'policy': 1.0, 'value': 1.0},
        metrics={
            'policy': [
                'categorical_accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc')
            ],
            'value': [
                'mse',
                keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                keras.metrics.AUC(name='auc')
            ]
        }
    )

    return model

# ------------------------------------------------------------------
# OneCycle Learning Rate Scheduler Callback.
class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(self, total_epochs, pct_start=0.3, base_lr=1e-4, max_lr=5e-4, final_lr=1e-5):
        """
        OneCycle learning rate scheduler:
         - In the first pct_start fraction of epochs, LR increases linearly from base_lr to max_lr.
         - In the remaining epochs, LR decays linearly from max_lr to final_lr.
        """
        super(OneCycleScheduler, self).__init__()
        self.total_epochs = total_epochs
        self.pct_start = pct_start
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.total_epochs * self.pct_start:
            # Linear increase phase
            increase_epochs = self.total_epochs * self.pct_start
            lr = self.base_lr + (self.max_lr - self.base_lr) * (epoch / increase_epochs)
        else:
            # Linear decay phase
            decay_epochs = self.total_epochs - self.total_epochs * self.pct_start
            epoch_in_decay = epoch - self.total_epochs * self.pct_start
            lr = self.max_lr - (self.max_lr - self.final_lr) * (epoch_in_decay / decay_epochs)
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        if logs is not None:
            logs["lr"] = lr

# ------------------------------------------------------------------
def plot_training_metrics(history_list, sample_points=50, save_path='results/training_metrics_balanced_v3.png'):
    """
    Plot training & validation metrics from the collected history.
    Shows loss, policy accuracy, and learning rate over epochs.
    """
    combined_history = {}
    for hist in history_list:
        for key, values in hist.items():
            combined_history.setdefault(key, []).extend(values)

    total_epochs = len(combined_history.get('loss', []))
    if total_epochs == 0:
        print("No training data to plot.")
        return
    step = max(1, total_epochs // sample_points)
    sampled_epochs = list(range(1, total_epochs + 1, step))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(sampled_epochs, [combined_history['loss'][i-1] for i in sampled_epochs],
             label='Train Loss', marker='o')
    if 'val_loss' in combined_history:
        ax1.plot(sampled_epochs, [combined_history['val_loss'][i-1] for i in sampled_epochs],
                 label='Val Loss', marker='o')
    if 'policy_categorical_accuracy' in combined_history:
        ax1.plot(sampled_epochs, [combined_history['policy_categorical_accuracy'][i-1] for i in sampled_epochs],
                 label='Train Policy Acc', marker='s')
    if 'val_policy_categorical_accuracy' in combined_history:
        ax1.plot(sampled_epochs, [combined_history['val_policy_categorical_accuracy'][i-1] for i in sampled_epochs],
                 label='Val Policy Acc', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss/Accuracy')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    if 'lr' in combined_history:
        ax2 = ax1.twinx()
        ax2.plot(sampled_epochs, [combined_history['lr'][i-1] for i in sampled_epochs],
                 label='Learning Rate', color='tab:gray', linestyle='--', marker='^')
        ax2.set_ylabel('Learning Rate')
        ax2.legend(loc='upper right')

    plt.title('BalancedGoNet Training Metrics')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
    print(f"Training metrics plot saved to {save_path}")

# ------------------------------------------------------------------
# save model
checkpoint_dir = "BalancedGoNet_checkpoints_v3_Adam"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# ------------------------------------------------------------------
model = create_balanced_model()
model.summary()  # Prints model architecture and parameter count

# Generate validation data (using golois to fill val_input, val_policy, val_value, val_end)
planes = 31
moves = 361
N = 10000  # Number of validation samples

val_input_data = np.random.randint(2, size=(N, 19, 19, planes)).astype('float32')
val_policy = np.random.randint(moves, size=(N,))
val_policy = keras.utils.to_categorical(val_policy, num_classes=moves)
val_value = np.random.randint(2, size=(N,)).astype('float32')
val_end = np.random.randint(2, size=(N, 19, 19, 2)).astype('float32')
val_groups = np.zeros((N, 19, 19, 1)).astype('float32')

print("Loading validation data...", flush=True)
golois.getValidation(val_input_data, val_policy, val_value, val_end)

# Data iterator for training.
data_loader = DataIterator(epochs=5, N=N)  # 5 epochs worth of data per iteration

# log metrics to a CSV file.
os.makedirs('results', exist_ok=True)
csv_logger = CSVLogger('results/training_log_balanced_v3_Adam.csv', append=True)

# Instantiate OneCycle LR Scheduler callback with tuned parameters
# Here total_epochs is set to 1000 (total global epochs across iterations)
total_epochs = 1000  
onecycle_lr = OneCycleScheduler(total_epochs=total_epochs, pct_start=0.3, base_lr=0.001, max_lr=5e-4, final_lr=1e-5)

# Early stopping parameters
best_iteration_loss = float('inf')
patience_counter = 0
# Early stopping will only be applied after global epoch > 333 (after decaying phase starts)
patience = 20

history_list = []  # to store history from each iteration
iteration = 0
over_epochs = 5    # 5 epochs per iteration (must match DataIterator.epochs)
max_iterations = 200  # Maximum iterations to train

global_epoch = 0  # to track total epochs so far

# Training loop over iterations
for inputs_data, policy_labels, value_labels, end_data, groups_data in data_loader:
    gc.collect()  # clear memory if needed
    iteration += 1
    history = model.fit(
        inputs_data,
        {'policy': policy_labels, 'value': value_labels},
        epochs=over_epochs,
        batch_size=256,  # Batch size selected to avoid GPU OOM
        validation_data=(val_input_data, [val_policy, val_value]),
        callbacks=[csv_logger, onecycle_lr],
        verbose=1
    )
    history_list.append(history.history)
    global_epoch += over_epochs
    avg_loss = np.mean(history.history['loss'][-2:])
    print(f"Iteration {iteration}: Global epoch {global_epoch}, Average training loss (last 2 epochs) = {avg_loss:.4f}")

    # Only start early stopping after global_epoch > 333
    if global_epoch > 333:
        if avg_loss < best_iteration_loss:
            best_iteration_loss = avg_loss
            patience_counter = 0
            best_model_path = "best_model_balanced_v3_Adam.h5"
            model.save(best_model_path)
            print(f"New best loss observed; model saved to '{best_model_path}'.")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} iteration(s).")

    if patience_counter >= patience:
        print("Early stopping triggered based on iteration loss.")
        break

    if iteration % 10 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration}.h5")
        model.save(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}.")

    if global_epoch >= total_epochs:
        print("Reached maximum global training epochs.")
        break

    del inputs_data, policy_labels, value_labels, end_data, groups_data

print(f"Training complete. CSV log saved to 'results/training_log_balanced_v3_Adam.csv'.")
plot_training_metrics(history_list, sample_points=50, save_path='results/training_metrics_balanced_v3_Adam.png')
