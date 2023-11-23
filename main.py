import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA

# taskA

X = np.load('C:/Users/45527/Desktop/MAL/XSound.npy')
Y = np.load('C:/Users/45527/Desktop/MAL/YSound.npy')

X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

Y_one_hot = to_categorical(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y_one_hot, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)


# taskB

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# taskC

pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train.reshape(X_train.shape[0], -1))
X_val_pca = pca.transform(X_val.reshape(X_val.shape[0], -1))
X_test_pca = pca.transform(X_test.reshape(X_test.shape[0], -1))

svm_model = SVC(kernel='linear')
svm_model.fit(X_train_pca, y_train.argmax(axis=1))

y_test_pred_svm = svm_model.predict(X_test_pca)

cm_svm = confusion_matrix(y_test.argmax(axis=1), y_test_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for SVM')
plt.show()

print(classification_report(y_test.argmax(axis=1), y_test_pred_svm))
