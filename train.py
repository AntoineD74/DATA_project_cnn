import transformer
import datasetProcessing as dp
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adadelta, Adam, Adamax, Lion, SGD


batch = next(iter(dp.valid_ds))

# The vocabulary to convert predicted indices into characters
idx_to_char = list(dp.char_to_num.keys())
display_cb = transformer.DisplayOutputs(batch, idx_to_char, target_start_token_idx=dp.char_to_num['<'], target_end_token_idx=dp.char_to_num['>'])

model = transformer.Transformer(
    num_hid=200,
    num_head=4,
    num_feed_forward=400,
    source_maxlen = dp.FRAME_LEN,
    target_maxlen=64,
    num_layers_enc=2,
    num_layers_dec=1,
    num_classes=62
)
loss_fn = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True, label_smoothing=0.1,
)

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

optimizers = [ Lion(), SGD()] #Adadelta, Lion Adamax(
optmizerNames = ["Lion", "SGD"]
results = []

for i in range(len(optimizers)):
    optimizer = optimizers[i]
    model.compile(optimizer=optimizer, loss=loss_fn)
        
    history = model.fit(dp.train_ds, validation_data=dp.valid_ds, callbacks=[display_cb], epochs=3, verbose=1)
    results.append(history)
    
plt.figure(figsize=(20, 15))
for j in range(len(optimizers)):
    plt.subplot(1, 2, 1)
    history = results[j]
    plt.plot(history.history['loss'], label=optmizerNames[j])
    leg = plt.legend(loc='upper right')
    plt.title("loss")
    plt.xlabel('Time')
    plt.ylabel('Accuracy')

for k in range(len(optimizers)):
    plt.subplot(1, 2, 2)
    history = results[k]
    plt.plot(history.history['val_loss'], label=optmizerNames[k])
    leg = plt.legend(loc='upper right')
    plt.title("Training loss")
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
   
plt.show()