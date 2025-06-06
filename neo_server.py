import argparse
import sys
import board
import neopixel
import stat
from time import sleep

SOCKET_PATH = '/tmp/neopixel_socket'
BYTES_PER_PIXEL = 3
LED_COUNT = 24
NUM_PIXELS = LED_COUNT // 3
LED_PIN = board.D18
pixels = neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=1, auto_write=False)

def handle_leds(data):
	#print(data)
	for i in range(0,8):
		pixel = (data[3*i],data[3*i + 1],data[3*i + 2])
		r,g,b = pixel

		r = min(16,r)
		g = min(16,g)
		b = min(16,b)

		pixel = (r,g,b)

		pixels[i] = pixel
		pixels[15-i] = pixel
		pixels[16+i] = pixel

	pixels.show()

import os
import socket
def main():
	if os.path.exists(SOCKET_PATH):
		os.remove(SOCKET_PATH)

	with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
		sock.bind(SOCKET_PATH)
		os.chmod(SOCKET_PATH, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
		sock.listen()

		while True:
			conn, addr = sock.accept()
			with conn:
				data = conn.recv(NUM_PIXELS * BYTES_PER_PIXEL)
				handle_leds(data)

if __name__ == "__main__":
	main()