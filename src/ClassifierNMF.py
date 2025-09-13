import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import argparse
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
import math
import wandb
from wandb.integration.keras.callbacks import WandbMetricsLogger, WandbModelCheckpoint
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay


parser = argparse.ArgumentParser(description="Train classifier on nmf transformed data")
parser.add_argument("--input_data", required=True, help="Path to input npy file.")
parser.add_argument("--input_lables",required=True, help="Input labels")
parser.add_argument("--output", required=True, help="Directory to save results.")
parser.add_argument("--nmf_encoder", required=True, help="NMF joblib")

parser.add_argument("--name", required=True, help="Base name for output files.")
args = parser.parse_args()



nmf = joblib.load(args.nmf_encoder)

data = np.load(args.input_data)
data = data.astype(np.float32)
Y = np.load(args.input_lables)
print(data.shape)

X = nmf.transform(data)
print(f'Shape of nmf transformed data: {X.shape}')

wandb.init(
    project="NMFClassifier",
    config={
        "hidden_size": 64,
        "activation": "relu",
        "dropout": 0.5,
        "output_activation": "softmax",
        "optimizer": "adam",
        "loss": "sparse_categorical_crossentropy",
        "metric": "accuracy",
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001
    }
)

config = wandb.config

class MLPClassifier(tf.keras.Model):
    def __init__(self, input_dim, num_classes=2, hidden_size=64, dropout_rate=0.5):
        super().__init__()
        self.hidden = layers.Dense(hidden_size, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.classifier = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.hidden(inputs)
        x = self.dropout(x)
        return self.classifier(x)
    
X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp)
print(f'Training data shape: {X_train.shape}')
print(f'Val data shape: {X_val.shape}')
print(f'Test data shape: {X_test.shape}')

input_dim = X_train.shape[1]
model = MLPClassifier(input_dim=input_dim,hidden_size=config.hidden_size)

early_stopping = EarlyStopping(
    monitor='accuracy',
    patience=3,
    restore_best_weights=True
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
            )

print("Training started....")
history = model.fit(
    X_train, y_train,
    epochs=50, batch_size=32,    
    validation_data=(X_val, y_val), 
    verbose=0,
    callbacks=[early_stopping,
               WandbMetricsLogger(),
               WandbModelCheckpoint("nmf_model_{epoch:02d}.keras",save_best_only=True,save_weights_only=False,monitor='accuracy')
               ]
)
wandb.finish()
print("Training completed!")


loss, accuracy = model.evaluate(X_test, y_test)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)


cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')

import matplotlib.pyplot as plt

if len(np.unique(y_test)) == 2:
    auc = roc_auc_score(y_test, y_pred_probs[:, 1])
    print(f'AUC: {auc}')
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    roc_path = os.path.join(args.output, f"{args.name}_roc_curve.png")
    plt.savefig(roc_path)
    plt.close()

# Confusion matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
cm_path = os.path.join(args.output, f"{args.name}_confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f'Test loss: {loss}')
print(f'Test Accuracy {accuracy}')
