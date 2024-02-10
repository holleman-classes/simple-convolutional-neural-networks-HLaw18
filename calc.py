import tensorflow as tf
from tabulate import tabulate
import matplotlib.pyplot as plt

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', input_shape=(32, 32, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10)
])

model.summary()

parameter_count = []
mac_count = []
output_size = []

for layer in model.layers:
    layer_params = tf.reduce_prod(tf.shape(layer.get_weights()[0]))
    parameter_count.append(layer_params)
    
    if isinstance(layer, tf.keras.layers.Conv2D):
        macs = layer_params * tf.reduce_prod(layer.kernel_size) * layer.input_shape[-1]
        mac_count.append(macs)
    else:
        mac_count.append(0)
    
    output_shape = layer.output_shape[1:] if isinstance(layer, tf.keras.layers.Conv2D) else layer.output_shape
    output_size.append(tf.reduce_prod(output_shape))

table_data = list(zip(parameter_count, mac_count, output_size))
table_headers = ["Layer", "Parameter Count", "MAC Count", "Output Size"]
table = tabulate(table_data, headers=table_headers, tablefmt="pretty")

with open("cnn_parameters_macs.pdf", "w") as f:
    f.write(table)

fig, ax = plt.subplots(figsize=(10, 6))
x = list(range(1, len(model.layers) + 1))

ax.bar(x, parameter_count, label="Parameter Count")
ax.bar(x, mac_count, label="MAC Count", alpha=0.7)
ax.bar(x, output_size, label="Output Size", alpha=0.5)

ax.set_xlabel("Layers")
ax.set_ylabel("Count")
ax.set_title("Parameter Count, MAC Count, and Output Size per Layer")
ax.legend()

plt.show()
