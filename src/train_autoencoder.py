import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import numpy as np
import pandas as pd
import tensorflow as tf
from models.autoencoder_model import build_autoencoder
from timer import start_timer 

DATA_DIR  = 'data/processed'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def load_unsup_train(path):
    df = pd.read_csv(path, compression='gzip')
    return df.values

def main():
    # load
    X_train = load_unsup_train(os.path.join(DATA_DIR, 'unsupervised_train_data.csv.gz'))
    input_dim = X_train.shape[1]

    # build & compile
    autoenc = build_autoencoder(input_dim=input_dim, bottleneck_dim=16)
    autoenc.compile(optimizer='adam', loss='mse')

    # callbacks → native .keras format
    ckpt_path = os.path.join(MODEL_DIR, 'autoencoder.keras')
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            save_best_only=True,
            monitor='val_loss'
        )
    ]

    # train
    autoenc.fit(
        X_train, X_train,
        epochs=50,
        batch_size=256,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=2
    )

    # batch‐wise threshold computation
    mse_batches = []
    for i in range(0, X_train.shape[0], 256):
        batch = X_train[i:i+256]
        rec = autoenc.predict(batch, batch_size=256, verbose=0)
        mse_batches.append(np.mean((batch - rec)**2, axis=1))
    mse = np.concatenate(mse_batches)
    threshold = np.percentile(mse, 99)

    # save threshold
    with open(os.path.join(MODEL_DIR, 'threshold.txt'), 'w') as f:
        f.write(f"{threshold:.6f}\n")

    print(f"\n✔ Training complete. Threshold set at {threshold:.6f}")
    print(f"✔ Model saved to {ckpt_path}")

if __name__ == '__main__':
    start_timer()
    print("\n")
    main()
