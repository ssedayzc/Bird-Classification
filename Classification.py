import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
class BirdClassification:
    def __init__(self, data_folder, batch_size=32):
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.features, self.labels = self._load_bird_data()
        self.current_index = 0
        
    def _load_bird_data(self):
        def batch_generator(images, labels, batch_size):
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                batch_features = list(map(lambda img_path: np.array(resize(rgb2gray(imread(img_path)), (224, 224))).flatten(), batch_images))
                yield np.array(batch_features), np.array(batch_labels)

        features = []
        labels = []

        for class_name in os.listdir(self.data_folder):
            class_folder = os.path.join(self.data_folder, class_name)
            if os.path.isdir(class_folder):
                images = [os.path.join(class_folder, filename) for filename in os.listdir(class_folder)]
                class_labels = [class_name] * len(images)
                for batch_features, batch_labels in batch_generator(images, class_labels, self.batch_size):
                    features.extend(batch_features)
                    labels.extend(batch_labels)

        return np.array(features), np.array(labels)


    def preprocess_data(self, data):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        return scaled_data

    def apply_preprocessing(self):
        self.features = self.preprocess_data(self.features)

    def data_generator(self):
        num_samples = len(self.features)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        while True:
            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                yield self.features[batch_indices], self.labels[batch_indices]

    def train_model(self, data_generator):
        model = KNeighborsClassifier(n_neighbors=2, metric='minkowski')
        X_batch, y_batch = next(data_generator)
        model.fit(X_batch, y_batch)
        return model

    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        return accuracy, precision, recall, f1


# Veri setinin bulunduğu klasör
data_folder = "Birds_Classification"

bird_classifier = BirdClassification(data_folder)

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(bird_classifier.features,
                                                    bird_classifier.labels,
                                                    test_size=0.2,
                                                    random_state=42)

# Modeli eğit
trained_model = bird_classifier.train_model(bird_classifier.data_generator())

# Modeli değerlendir
accuracy, precision, recall, f1 = bird_classifier.evaluate_model(trained_model, X_test, y_test)
print("Model Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
