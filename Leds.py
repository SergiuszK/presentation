import socket

SOCKET_PATH = "/tmp/neopixel_socket"


class Leds:
    RED = 0
    GREEN = 1
    BLUE = 2
    BLACK = 4

    def __init__(self):
        self.set_leds([self.BLACK, self.BLACK, self.BLACK, self.BLACK, self.BLACK, self.BLACK, self.BLACK, self.BLACK])

    def send_led_data(self, led_data):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(SOCKET_PATH)
            s.sendall(led_data)

    def set_leds(self, colors):
        led_data = bytearray(8 * 3)
        for x in range(0,24):
            led_data[x] = 0

        a = 0
        b = 3
        for color in colors:
            for x in range(a,b):
                if color == Leds.BLACK:
                    break
                if x % 3 == color:
                    led_data[x] = 4
            a+=3
            b+=3

        self.send_led_data(led_data)
