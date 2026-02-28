import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv("../data/bitcoin.csv")

# Удаление лишнего столпца
data = data.drop(['Date'], axis=1)

# Разделяем на фичи
X = data.drop(['PriceCategory'], axis=1)
y = data[['PriceCategory']]


# OHE для "y" фичи
encoder = OneHotEncoder(sparse_output=False)
y_ohe = encoder.fit_transform(y)
y_features_name = encoder.get_feature_names_out(['PriceCategory'])
converted_y = pd.DataFrame(y_ohe, columns=y_features_name)

# Добавление OHE столпцов в датасет
data = pd.concat([data, converted_y], axis=1)

# Train, test split
X_train, X_test, y_train, y_test = train_test_split(X, y_ohe, test_size=0.2, random_state=42)

# StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Основная модель
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)), # Hidden layer 1
    # Dropout(0.3),
    Dense(32, activation='relu'), # Hidden layer 2
    # Dropout(0.2),

    Dense(3, activation='softmax') # O/p layer
])

# Optimizer, loss
opt=tf.keras.optimizers.Adam(learning_rate=0.01)
loss=tf.keras.losses.CategoricalCrossentropy()

# EarlyStopping
early_stopping=EarlyStopping(monitor='val_loss', patience=5)

# Фитаем модель
model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, validation_data=[X_test, y_test], callbacks=[early_stopping])

# Сохраняем
model.save("bitcoin.h5")