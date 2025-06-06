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

model = build_model()

model.load_weights('../../models/cnn_6561_outputs_final_250.h5')

def get_last_conv_layer_name(model):
    """Znajduje nazwę ostatniej warstwy konwolucyjnej w modelu."""
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("Could not find a convolution layer in the model.")

LAST_CONV_LAYER_NAME = get_last_conv_layer_name(model)

# --- Funkcje do Grad-CAM (bez zmian) ---
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

# --- NOWE FUNKCJE DO WIZUALIZACJI PREDYKCJI ---
def index_to_combination(index, num_segments=8, base=3):
    """Konwertuje indeks klasy (0-6560) na 8-elementową listę kolorów."""
    if not 0 <= index < base**num_segments:
        raise ValueError(f"Index must be between 0 and {base**num_segments - 1}")
    
    combination = []
    temp_index = index
    for _ in range(num_segments):
        combination.insert(0, temp_index % base)
        temp_index //= base
    return combination

def draw_prediction_strip(combination, segment_height=40, segment_width=80):
    """Tworzy obrazek z pionowym paskiem kolorów reprezentującym predykcję."""
    # Definicja kolorów (R, G, B) - Matplotlib używa RGB
    color_map = {
        0: (255, 0, 0),    # Czerwony
        1: (0, 255, 0),    # Zielony
        2: (0, 0, 255)     # Niebieski
    }
    
    strip_height = segment_height * len(combination)
    strip = np.zeros((strip_height, segment_width, 3), dtype=np.uint8)
    
    for i, color_code in enumerate(combination):
        start_y = i * segment_height
        end_y = (i + 1) * segment_height
        color_rgb = color_map.get(color_code, (0, 0, 0)) # Domyślnie czarny
        strip[start_y:end_y, :] = color_rgb
        
    return strip

def load_and_preprocess_image(image, input_shape=(256, 256)):
    """Przetwarza klatkę przechwyconą z wideo."""
    image = cv2.resize(image, input_shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_for_display = image.copy()
    
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image, image_for_display

# --- Konfiguracja klienta ---
HOST = '192.168.1.100'  # Zastąp adresem IP swojego Raspberry Pi
PORT = 8089

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client_socket.connect((HOST, PORT))
    print(f"Connected to {HOST}:{PORT}")
except ConnectionRefusedError:
    print(f"Connection to {HOST}:{PORT} was refused. Make sure the server script is running on the Raspberry Pi.")
    exit()


data = b""
payload_size = struct.calcsize("L")
cv2.namedWindow('Live Stream')

while True:
    try:
        while len(data) < payload_size:
            data += client_socket.recv(4096)
            
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]
        
        while len(data) < msg_size:
            data += client_socket.recv(4096)
            
        frame_data = data[:msg_size]
        data = data[msg_size:]
        
        frame = pickle.loads(frame_data)
        
        cv2.imshow('Live Stream', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 13: # Enter
            print("Enter pressed. Processing frame...")
            
            processed_image, display_image = load_and_preprocess_image(frame)
            
            predictions = model.predict(processed_image)
            predicted_class_index = np.argmax(predictions[0])
            
            # 1. Konwersja indeksu na kombinację kolorów
            predicted_combination = index_to_combination(predicted_class_index)
            print(f"Predicted Class Index: {predicted_class_index}")
            print(f"Predicted Combination: {predicted_combination}")

            # 2. Generowanie heatmapy Grad-CAM
            heatmap = make_gradcam_heatmap(processed_image, model, LAST_CONV_LAYER_NAME, pred_index=predicted_class_index)
            overlayed_image = overlay_heatmap(display_image, heatmap)
            
            # 3. Tworzenie wizualizacji predykcji
            prediction_strip = draw_prediction_strip(predicted_combination)
            
            # 4. Wyświetlanie wyników w nowym, rozbudowanym oknie
            fig, axs = plt.subplots(1, 4, figsize=(18, 5))
            
            axs[0].imshow(display_image)
            axs[0].set_title("Original Frame")
            axs[0].axis("off")

            axs[1].imshow(heatmap, cmap="viridis")
            axs[1].set_title("Grad-CAM Heatmap")
            axs[1].axis("off")

            axs[2].imshow(overlayed_image)
            axs[2].set_title("Overlayed Image")
            axs[2].axis("off")
            
            axs[3].imshow(prediction_strip)
            axs[3].set_title("Prediction")
            axs[3].axis("off")
            
            fig.suptitle(f"Predicted Combination: {predicted_combination}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Dostosowanie, by tytuł główny nie nachodził
            plt.show()

        if key == ord('q'):
            break
            
    except (ConnectionResetError, BrokenPipeError):
        print("Connection to server lost. Exiting.")
        break
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        break

cv2.destroyAllWindows()
client_socket.close()