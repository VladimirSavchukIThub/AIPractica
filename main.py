import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
#
# НЕ РАСКОМЕНЧИВАТЬ
# model = tf.keras.models.Sequential([])
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation="relu"))
# model.add(tf.keras.layers.Dense(128, activation="relu"))
# model.add(tf.keras.layers.Dense(10, activation="softmax"))
#
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#
# model.fit(x_train, y_train, epochs=50)
#
# model.save('handwritten.model')
# test_loss, test_accuracy = model.evaluate(x_test, y_test)
# print(f"Точность на тестовых данных: {test_accuracy}")

#model = tf.keras.models.load_model('handwritten.model')

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dense(36, activation="softmax")  # 10 цифр + 26 букв
# ])
#
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#
# model.fit(x_train, y_train, epochs=50)
# model.save('handwritten.model')




model = tf.keras.models.load_model('handwritten.model')

# НЕ РАСКОМЕНЧИВАТЬ
# image_number = 1
# while os.path.isfile(f"digits/digit{image_number}.png"):
#     try:
#         img = cv2.imread(f"digits/digit{image_number}.png")[:, :, 0]
#         img = np.invert(np.array([img]))
#         prediction = model.predict(img)
#         predicted_digit = np.argmax(prediction)
#
#         plt.imshow(img[0], cmap=plt.cm.binary)
#         plt.show()
#
#         feedback = input(f"Это цифра: {predicted_digit}. Это правильно? (да/нет): ")
#
#         if feedback.lower() == 'нет':
#             # Принятое предсказание считается неправильным, обновляем данные для переобучения модели
#             correct_label = int(input("Введите правильную метку для этой цифры (0-9): "))
#             y_train_updated = np.array([correct_label])
#             x_train_updated = np.array([img.reshape(28, 28)])  # При необходимости, измените форму массива на нужную
#
#             # Если x_train и y_train еще не определены, определите их здесь
#             try:
#                 x_train = np.concatenate((x_train, x_train_updated))
#                 y_train = np.concatenate((y_train, y_train_updated))
#             except NameError:
#                 x_train = x_train_updated
#                 y_train = y_train_updated
#
#             # Переобучение модели
#             model.fit(x_train, y_train, epochs=10)
#
#             # Сохранение обновленной модели
#             model.save('handwritten.model')
#
#         elif feedback.lower() == 'да':
#             # Принятое предсказание считается правильным, переходим к следующему изображению без изменения модели
#             pass
#
#     except Exception as e:
#         print(f"Ошибка: {e}")
#
#     finally:
#         image_number += 1

image_number = 1

# Словарь меток
labels_dict = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}

while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)

        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

        predicted_symbol = labels_dict.get(predicted_class, 'Неизвестно')

        print(f"Предсказанный символ: {predicted_symbol}")

        feedback = input("Это правильный символ? (да/нет): ")

        if feedback.lower() == 'нет':
            correct_label = int(input("Введите правильную метку для этого символа (0-35): "))
            y_train_updated = np.array([correct_label])
            x_train_updated = np.array([img.reshape(28, 28)])

            try:
                x_train = np.concatenate((x_train, x_train_updated))
                y_train = np.concatenate((y_train, y_train_updated))
            except NameError:
                x_train = x_train_updated
                y_train = y_train_updated

            model.fit(x_train, y_train, epochs=1)
            model.save('handwritten.model')

        elif feedback.lower() == 'да':
            pass

    except Exception as e:
        print(f"Ошибка: {e}")

    finally:
        image_number += 1