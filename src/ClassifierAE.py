import numpy as np
import tensorflow as tf
import argparse
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

parser = argparse.ArgumentParser(description="Train classifier on top of encoder")
parser.add_argument("--x", required=True, help="Path to the data.")
parser.add_argument("--y", required=True, help="Path to the labels")
parser.add_argument("--output", required=True, help="Directory to save the classiier.")
parser.add_argument("--name", required=True, help="Name to save output")
parser.add_argument("--encoder", required=True, help="Path to the existing encoder .keras file")

args = parser.parse_args()


encoder = load_model(args.encoder)
X = np.load(args.x)
Y = np.load(args.y)
print(f'Shape of input data: {X.shape}')

class MLPClassifier(tf.keras.Model):
    def __init__(self, encoder, num_classes=2, hidden_size=64, dropout_rate=0.5):
        super().__init__()
        self.encoder = encoder
        self.encoder.trainable = False  # Freeze encoder
        self.hidden = layers.Dense(hidden_size, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.classifier = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.hidden(x)
        x = self.dropout(x)
        return self.classifier(x)

# Split into train (70%), val (10%) and test (20%) 
X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp)

print(f'Training data shape: {X_train.shape}')
print(f'Val data shape: {X_val.shape}')
print(f'Test data shape: {X_test.shape}')

model = MLPClassifier(encoder)

early_stopping = EarlyStopping(
    monitor='accuracy',
    patience=3,
    restore_best_weights=True
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
            )

model.summary()

print("Training started....")
history = model.fit(
    X_train, y_train,
    epochs=50, batch_size=32,    
    validation_data=(X_val, y_val), 
    verbose=0,
    callbacks=[early_stopping]
)

# Plot and save training graphs
import matplotlib.pyplot as plt

# Accuracy and Loss plot on the same axis
fig, ax1 = plt.subplots()

# Plot accuracy
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy', color='tab:blue')
ax1.plot(history.history['accuracy'], label='Train Accuracy', color='tab:blue', linestyle='-')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='tab:blue', linestyle='--')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.legend(loc='upper left')

# Plot loss on the same axis
ax2 = ax1.twinx()
ax2.set_ylabel('Loss', color='tab:orange')
ax2.plot(history.history['loss'], label='Train Loss', color='tab:orange', linestyle='-')
ax2.plot(history.history['val_loss'], label='Validation Loss', color='tab:orange', linestyle='--')
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax2.legend(loc='upper right')

plt.title('Training and Validation Accuracy and Loss')
combined_plot_path = os.path.join(args.output, f"{args.name}_combined_plot.png")
plt.savefig(combined_plot_path)
print(f"Combined accuracy and loss plot saved to {combined_plot_path}")
plt.close()

print("Training completed!")

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

loss, accuracy = model.evaluate(X_test, y_test)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)


cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
specificity = tn / (tn + fp)
false_positive_rate = fp / (fp + tn)
false_negative_rate = fn / (fn + tp)
true_positive_rate = tp / (tp + fn)  # Same as recall/sensitivity
negative_predictive_value = tn / (tn + fn)

print(f'Precision: {precision:.4f}')
print(f'Recall (Sensitivity/TPR): {recall:.4f}')
print(f'Specificity (TNR): {specificity:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'False Positive Rate (FPR): {false_positive_rate:.4f}')
print(f'False Negative Rate (FNR): {false_negative_rate:.4f}')
print(f'Negative Predictive Value (NPV): {negative_predictive_value:.4f}')


if len(np.unique(y_test)) == 2:
    auc = roc_auc_score(y_test, y_pred_probs[:, 1])
    print(f'AUC: {auc}')
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs[:, 1])
    import matplotlib.pyplot as plt
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    roc_curve_path = os.path.join(args.output, f"{args.name}_roc_curve.png")
    plt.savefig(roc_curve_path)
    print(f"ROC curve saved to {roc_curve_path}")
    plt.close()

# Confusion matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
conf_matrix_path = os.path.join(args.output, f"{args.name}_conf_matrix.png")
plt.savefig(conf_matrix_path)
print(f"Confusion matrix saved to {conf_matrix_path}")
plt.close()
print(f'Test loss: {loss}')
print(f'Test Accuracy {accuracy}')

save_path = os.path.join(args.output, f"{args.name}.keras")
model.save(save_path)
print(f"Classifier trained and saved to {save_path}")
