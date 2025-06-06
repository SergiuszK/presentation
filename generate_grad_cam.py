import cv2
import numpy as np
import socket
import pickle
import struct
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models


def build_model(input_shape=(256, 256, 3), num_classes=6561):
    model = models.Sequential([
        # Base layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.25),

        layers.Dense(num_classes, activation='softmax')
    ])

    return model

new_model = build_model()

new_model.load_weights('../../models/cnn_6561_outputs_final_250.h5')

def get_last_conv_layer_name(model):
    """Znajduje nazwę ostatniej warstwy konwolucyjnej w modelu."""
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("Could not find a convolution layer in the model.")

LAST_CONV_LAYER_NAME = get_last_conv_layer_name(model)

# --- Funkcje do Grad-CAM (bez zmian z Twojego kodu) ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap, alpha=0.7, cmap="viridis"):
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    
    colormap = plt.get_cmap(cmap)
    colored_heatmap = colormap(heatmap_resized)[:, :, :3]
    colored_heatmap = np.uint8(colored_heatmap * 255)
    
    overlayed_img = cv2.addWeighted(original_img, 1 - alpha, colored_heatmap, alpha, 0)
    return overlayed_img


def load_and_preprocess_image(image, input_shape=(256, 256)):
    """Przetwarza klatkę przechwyconą z wideo."""
    image = cv2.resize(image, input_shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV wczytuje jako BGR
    image_for_display = image.copy() # Kopia do wyświetlenia
    
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image, image_for_display

# --- Konfiguracja klienta ---
HOST = '192.168.1.100'  # Zastąp adresem IP swojego Raspberry Pi
PORT = 8089

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
print(f"Connected to {HOST}:{PORT}")

data = b""
payload_size = struct.calcsize("L")

cv2.namedWindow('Live Stream')

while True:
    while len(data) < payload_size:
        data += client_socket.recv(4096)
        
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]
    
    while len(data) < msg_size:
        data += client_socket.recv(4096)
        
    frame_data = data[:msg_size]
    data = data[msg_size:]
    
    # Deserializacja ramki
    frame = pickle.loads(frame_data)
    
    cv2.imshow('Live Stream', frame)
    key = cv2.waitKey(1) & 0xFF

    # Naciśnięcie Enter (kod 13)
    if key == 13:
        print("Enter pressed. Processing frame...")
        
        # 1. Przetwarzanie obrazu
        processed_image, display_image = load_and_preprocess_image(frame)

        # 2. Predykcja
        prediction = model.predict(processed_image)
        # Możesz przetworzyć `prediction` aby uzyskać czytelną klasę
        print(f"Prediction: {prediction[0]}")

        # 3. Generowanie heatmapy Grad-CAM
        heatmap = make_gradcam_heatmap(processed_image, model, LAST_CONV_LAYER_NAME)
        overlayed_image = overlay_heatmap(display_image, heatmap)
        
        # 4. Wyświetlanie wyników w nowym oknie
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 3, 1)
        plt.title("Original Frame")
        plt.imshow(display_image)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Grad-CAM Heatmap")
        plt.imshow(heatmap, cmap="viridis")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Overlayed Image")
        plt.imshow(overlayed_image)
        plt.axis("off")
        
        plt.suptitle(f"Prediction Result: {prediction[0]}")
        plt.tight_layout()
        plt.show()

    # Naciśnięcie 'q' aby zakończyć
    if key == ord('q'):
        break

cv2.destroyAllWindows()
client_socket.close()