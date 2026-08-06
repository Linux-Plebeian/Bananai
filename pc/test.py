import numpy as np
import json
import imgtoarray as ita
import trainer as tr
import asdlkfj as deez
import camera as c
import neural_network as nn
from PIL import Image
print(deez.nuts())

failed = 0
total = 15
for i in range(1, 5):
    ripe = ita.convert_rgb(f"test_images/test_ripe{i}.jpg")
    unripe = ita.convert_rgb(f"test_images/test_unripe{i}.jpg")
    overripe = ita.convert_rgb(f"test_images/test_overripe{i}.jpg")


    test_unripe = np.round(nn.forward_prop(unripe, nn.weights1, nn.bias1, nn.weights2, nn.bias2, nn.weights3, nn.bias3, nn.weights4, nn.bias4, nn.weights5, nn.bias5), decimals=3)
    test_ripe = np.round(nn.forward_prop(ripe, nn.weights1, nn.bias1, nn.weights2, nn.bias2, nn.weights3, nn.bias3, nn.weights4, nn.bias4, nn.weights5, nn.bias5), decimals=3)
    test_overripe = np.round(nn.forward_prop(overripe, nn.weights1, nn.bias1, nn.weights2, nn.bias2, nn.weights3, nn.bias3, nn.weights4, nn.bias4, nn.weights5, nn.bias5), decimals=3)

    print("========================")
    print(f"TEST UNRIPE {i}")
    print(test_unripe)
    if test_unripe[0][0] > test_unripe[0][1] and test_unripe[0][0] > test_unripe[0][2]:
        print("===PASS===")
    else:
        print("===FAIL===")
        failed += 1

    print(f"TEST RIPE {i}")
    print(test_ripe)
    if test_ripe[0][1] > test_ripe[0][2] and test_ripe[0][1] > test_ripe[0][0]:
        print("===PASS===")
    else:
        print("===FAIL===")
        failed += 1
    print(f"TEST OVERRIPE {i}")
    print(test_overripe)
    if test_overripe[0][2] >= test_overripe[0][0] and test_overripe[0][2] > test_overripe[0][1]:
        print("===PASS===")
    else:
        print("===FAIL===")
        failed += 1

    print("========================\n")

print("====", total-failed, " / ", total, "PASSED ===")
print("===", np.round((total-failed)*100/total, 3), "% PASS RATE ==")