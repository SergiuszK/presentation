import cv2
import numpy as np
import socket
import pickle
import struct
import tensorflow as tf
from tensorflow.keras import layers, models
import traceback

# --- Definicje funkcji i modelu (bez zmian) ---
def build_model(input_shape=(256, 256, 3), num_classes=6561):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', name="last_conv_layer"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("Could not find a convolution layer in the model.")

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
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

def index_to_combination(index, num_segments=8, base=3):
    if not 0 <= index < base**num_segments:
        return [0] * num_segments
    combination = []
    temp_index = index
    for _ in range(num_segments):
        combination.insert(0, temp_index % base)
        temp_index //= base
    return combination

def apply_gradcam_overlay(original_img_bgr, heatmap, alpha=0.5):
    heatmap_resized = cv2.resize(heatmap, (original_img_bgr.shape[1], original_img_bgr.shape[0]))
    heatmap_8bit = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
    overlayed_img = cv2.addWeighted(original_img_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return overlayed_img

# --- Główna część skryptu ---
print("Loading model...")
model = build_model()
try:
    model.load_weights('/home/user/presentation/presentation/cnn_6561_outputs_final_294.h5')
    print("Weights loaded successfully.")
except Exception as e:
    print(f"FATAL: Error loading weights: {e}")
    exit()

LAST_CONV_LAYER_NAME = get_last_conv_layer_name(model)

HOST = '10.173.140.249'
PORT = 8089

print(f"Connecting to {HOST}:{PORT}...")
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client_socket.connect((HOST, PORT))
    print("Connected.")
except Exception as e:
    print(f"FATAL: Connection failed: {e}")
    exit()

data = b""
payload_size = struct.calcsize("L")

while True:
    try:
        # --- Odbiór klatki ---
        while len(data) < payload_size: data += client_socket.recv(4096)
        packed_msg_size, data = data[:payload_size], data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]
        while len(data) < msg_size: data += client_socket.recv(4096)
        frame_data, data = data[:msg_size], data[msg_size:]
        original_frame = pickle.loads(frame_data)
        
        # --- KLUCZOWA ZMIANA ---
        # Sprawdzamy, czy obraz ma 4 kanały (BGRA) i konwertujemy go na 3 kanały (BGR)
        # To gwarantuje, że wymiary (w tym liczba kanałów) będą zawsze zgodne.
        if original_frame.shape[2] == 4:
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGRA2BGR)
        # --- KONIEC KLUCZOWEJ ZMIANY ---

        # --- Przygotowanie obrazów ---
        display_size = (640, 640)
        left_panel = cv2.resize(original_frame, display_size)
        right_panel_base = left_panel.copy()
        
        model_input_image = cv2.resize(original_frame, (256, 256))
        model_input_image_rgb = cv2.cvtColor(model_input_image, cv2.COLOR_BGR2RGB)
        model_input_image_norm = np.expand_dims(model_input_image_rgb.astype(np.float32) / 255.0, axis=0)
        
        # --- Predykcja i Grad-CAM ---
        predictions = model.predict(model_input_image_norm, verbose=0)
        predicted_class_index = np.argmax(predictions[0])
        heatmap = make_gradcam_heatmap(model_input_image_norm, model, LAST_CONV_LAYER_NAME, pred_index=predicted_class_index)
        
        # --- Tworzenie prawego panelu ---
        right_panel = apply_gradcam_overlay(right_panel_base, heatmap)
        
        predicted_combination = index_to_combination(predicted_class_index)
        prediction_text = f"Pred: {predicted_combination}"
        
        cv2.rectangle(right_panel, (0, display_size[1] - 30), (display_size[0], display_size[1]), (0,0,0), -1)
        cv2.putText(right_panel, prediction_text, (10, display_size[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # --- Łączenie paneli i wyświetlanie ---
        final_image = np.hstack((left_panel, right_panel))
        final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Real-Time Grad-CAM Analysis', final_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    except (ConnectionResetError, BrokenPipeError):
        print("Connection to server lost. Exiting.")
        break
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
        traceback.print_exc()
        break

print("Closing application.")
cv2.destroyAllWindows()
client_socket.close()