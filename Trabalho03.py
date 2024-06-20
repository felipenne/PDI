import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers, models

# Passo 1: Carregamento dos Dados
data = np.loadtxt(r'c:\Users\usuario\Downloads\ocr_car_numbers_rotulado.txt')
X = data[:, :-1]
y = data[:, -1]

# Passo 2: Divisão dos Dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Passo 3: Definição dos Extratores de Atributos
def extract_hog_features(images):
    hog_features = []
    for image in images:
        hog_feature = hog(image.reshape((35, 35)), pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        hog_features.append(hog_feature)
    return np.array(hog_features)

def extract_lbp_features(images):
    lbp_features = []
    for image in images:
        lbp_feature = local_binary_pattern(image.reshape((35, 35)), 8, 1, method='default')
        lbp_features.append(lbp_feature.ravel())
    return np.array(lbp_features)

pca = PCA(n_components=50)  # Reduzindo para 50 componentes principais
pca.fit(X_train)
def extract_pca_features(images):
    return pca.transform(images)

# Passo 4: Definição dos Classificadores de Padrão
svm_classifier = SVC(kernel='linear')

def build_cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(35, 35, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Função para converter dados para formato de entrada da CNN
def preprocess_for_cnn(X):
    return X.reshape(-1, 35, 35, 1)

# Passo 5: Avaliação dos Classificadores
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Avaliar SVM
svm_accuracies = cross_val_score(svm_classifier, X_train, y_train, cv=kfold)
svm_mean_accuracy = np.mean(svm_accuracies)
svm_std_accuracy = np.std(svm_accuracies)
print("SVM Mean Accuracy:", svm_mean_accuracy)
print("SVM Standard Deviation:", svm_std_accuracy)

# Avaliar CNN
cnn_accuracies = []
for train_index, test_index in kfold.split(X_train):
    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
    
    cnn_model = build_cnn()
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(preprocess_for_cnn(X_train_fold), y_train_fold, epochs=10, batch_size=64, verbose=0)
    cnn_accuracy = cnn_model.evaluate(preprocess_for_cnn(X_test_fold), y_test_fold, verbose=0)[1]
    cnn_accuracies.append(cnn_accuracy)

cnn_mean_accuracy = np.mean(cnn_accuracies)
cnn_std_accuracy = np.std(cnn_accuracies)
print("CNN Mean Accuracy:", cnn_mean_accuracy)
print("CNN Standard Deviation:", cnn_std_accuracy)

# Treinar os classificadores SVM e CNN com todo o conjunto de treinamento
svm_classifier.fit(X_train, y_train)
cnn_model = build_cnn()
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(preprocess_for_cnn(X_train), y_train, epochs=10, batch_size=64, verbose=0)

# Avaliar desempenho nos dados de teste
svm_test_predictions = svm_classifier.predict(X_test)
svm_test_accuracy = accuracy_score(y_test, svm_test_predictions)
print("SVM Test Accuracy:", svm_test_accuracy)

cnn_test_accuracy = cnn_model.evaluate(preprocess_for_cnn(X_test), y_test, verbose=0)[1]
print("CNN Test Accuracy:", cnn_test_accuracy)

# Matriz de confusão para SVM
svm_conf_matrix = confusion_matrix(y_test, svm_test_predictions)
print("SVM Confusion Matrix:")
print(svm_conf_matrix)

# Matriz de confusão para CNN
cnn_test_predictions = np.argmax(cnn_model.predict(preprocess_for_cnn(X_test)), axis=1)
cnn_conf_matrix = confusion_matrix(y_test, cnn_test_predictions)
print("CNN Confusion Matrix:")
print(cnn_conf_matrix)
