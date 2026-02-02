import tensorflow as tf
import numpy as np

# 1. Menyiapkan Data (Contoh: Hubungan y = 2x - 1)
# AI akan mencoba menebak rumus ini tanpa kita beri tahu
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# 2. Arsitektur Model (Otak AI)
# Layer 'Dense' dengan 1 neuron adalah bentuk paling sederhana
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# 3. Kompilasi Model
# Optimizer 'sgd' membantu AI belajar dari kesalahan (loss)
model.compile(optimizer='sgd', loss='mean_squared_error')

# 4. Melatih AI (Training)
# AI akan melihat data sebanyak 500 kali untuk mencari pola
print("Sedang melatih AI...")
model.fit(xs, ys, epochs=500, verbose=0)

# 5. Menggunakan AI untuk Prediksi
print("Hasil prediksi untuk angka 10 adalah:")
print(model.predict(np.array([10.0])))
