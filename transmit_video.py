import cv2
from picamera2 import Picamera2
from libcamera import Transform
import socket
import pickle
import struct

# Inicjalizacja kamery
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1280, 720)}, transform=Transform(hflip=1, vflip=1))
picam2.configure(config)
picam2.start()

# Konfiguracja serwera
HOST = '10.173.140.61'  # Użyj adresu IP Raspberry Pi
PORT = 8089

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(f"Socket created at {HOST}:{PORT}")
server_socket.bind((HOST, PORT))
server_socket.listen(10)
print("Socket now listening")

conn, addr = server_socket.accept()
print(f"Connection from: {addr}")

data = b""
payload_size = struct.calcsize("L")

while True:
    try:
        frame = picam2.capture_array()
        
        # Serializacja ramki
        a = pickle.dumps(frame)
        message = struct.pack("L", len(a)) + a
        
        # Wysłanie ramki
        conn.sendall(message)
        
    except (BrokenPipeError, ConnectionResetError):
        print("Client disconnected. Waiting for a new connection.")
        conn, addr = server_socket.accept()
        print(f"Connection from: {addr}")
    except Exception as e:
        print(f"An error occurred: {e}")
        break

print("Stopping server.")
conn.close()
picam2.stop()