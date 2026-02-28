import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv('../data/Iris.csv')

# Удаление не нужных полей
data=data.drop(['Id'], axis=1)

# X и y фичи
X = data.drop(["Species"], axis=1)
y = data[["Species"]]

# OHE "y" фичи
encoder = OneHotEncoder(sparse_output=False)
y_ohe = encoder.fit_transform(y)
features_name = encoder.get_feature_names_out(['Species'])
converted_encoder = pd.DataFrame(y_ohe, columns=features_name)

# Добавление OHE колонок в датасет
data = pd.concat([data, converted_encoder], axis=1)

# train, test split
X_train, X_test, y_train, y_test = train_test_split(X, y_ohe, test_size=0.2, random_state=42)

# Основная модель
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)), # Hidden layer 1
    Dense(32, activation='relu'), # Hidden layer 2

    Dense(3, activation='softmax') # Output layer with 3 classes
])

# Оптимизатор, лосс-функция и early stopping
opt=tf.keras.optimizers.Adam(learning_rate=0.01)
loss=tf.keras.losses.CategoricalCrossentropy()
early_stopping_callback=EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

# Фит модели
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping_callback])

# Сохранение модели
model.save("iris.h5")

print(model.summary())
# ------------------------------------
#  Total params: 7,499 (29.30 KB)
#  Trainable params: 2,499 (9.76 KB)
#  Non-trainable params: 0 (0.00 B)
#  Optimizer params: 5,000 (19.54 KB)
# ------------------------------------