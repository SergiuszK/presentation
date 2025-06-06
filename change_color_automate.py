from Leds import Leds
import time
import paramiko

start_combination = [0, 0, 0, 0, 0, 0, 0, 0]

n=8
colors = [0, 0, 0, 0, 0, 0, 0, 0]
black = [4,4,4,4,4,4,4,4]

leds = Leds()

def combination_to_index(combination, max_value):
    index = 0
    for value in combination:
        index = index * max_value + value
    return index

max_value = 3

start_index = combination_to_index(start_combination, max_value)

table = start_combination[:]

total_combinations = max_value ** n

for _ in range(start_index, total_combinations):
    leds.set_leds(colors)
    time.sleep(10)

    for i in range(n - 1, -1, -1):
        colors[i] += 1 
        if colors[i] < max_value:
            break 
        colors[i] = 0 