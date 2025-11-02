import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import argparse
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping



parser = argparse.ArgumentParser(description="Train classifier on top of encoder")
parser.add_argument("--x", required=True, help="Path to the data.")
parser.add_argument("--y", required=True, help="Path to the labels")
parser.add_argument("--output", required=True, help="Directory to save the classiier.")
parser.add_argument("--name", required=True, help="Name to save output")
parser.add_argument("--encoder", required=True, help="Path to the existing encoder .keras file")
parser.add_argument("--scaler", required=True, help="Path to the existing scaler")


args = parser.parse_args()


pca = joblib.load(args.encoder)
pca_scaler = joblib.load(args.scaler)
data = np.load(args.x)
Y = np.load(args.y)



# data = data.astype(np.float32)
print(f"Data before transform: {data.shape}")

X = pca_scaler.fit_transform(data)
X = pca.fit_transform(X)
print(X.shape)
print(f"Transformed  data: {X.shape}")


class MLPClassifier(tf.keras.Model):
    def __init__(self, num_classes=2, hidden_size=64, dropout_rate=0.5):
        super().__init__()
        self.hidden = layers.Dense(hidden_size, activation='tanh')
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

model = MLPClassifier()

early_stopping = EarlyStopping(
    monitor='accuracy',
    patience=3,
    restore_best_weights=True
)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0015,
    decay_steps=500,
    decay_rate=0.96,
    staircase=True
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
            )

print("Training started....")
history = model.fit(
    X_train, y_train,
    epochs=50, batch_size=32,    
    validation_data=(X_val, y_val), 
    verbose=0,
)
# wandb.finish()
print("Training completed!")

model.summary()

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay

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
    roc_path = os.path.join(args.output, f"{args.name}_roc_c_pca_curve.png")
    plt.savefig(roc_path)
    plt.close()

# Confusion matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
cm_path = os.path.join(args.output, f"{args.name}_confusion_pca_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f'Test loss: {loss}')
print(f'Test Accuracy {accuracy}')
