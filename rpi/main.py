import numpy as np
import json
import imgtoarray as ita
import asdlkfj as deez
import camera as c
import oled
import neural_network as nn
import time

print(deez.nuts())



oled.test()
print("Initializing . . .")
oled.oled.text("Initializing . . .", 0, 10,1)
oled.oled.show()
while True:
	#path = input("Enter image filename: ")
	print("Starting . . .")
	oled.oled.text("Starting . . .",0,19,1)
	oled.oled.show()
	while True:
		c.main()
		input_image1 = ita.convert_rgb(f"images/banana1.png")
		input_image2 = ita.convert_rgb(f"images/banana2.png")
		input_image3 = ita.convert_rgb(f"images/banana3.png")
		prediction1 = np.round(nn.forward_prop(input_image1, nn.weights1, nn.bias1, nn.weights2, nn.bias2, nn.weights3, nn.bias3, nn.weights4, nn.bias4, nn.weights5, nn.bias5), decimals=3)
		prediction2 = np.round(nn.forward_prop(input_image2, nn.weights1, nn.bias1, nn.weights2, nn.bias2, nn.weights3, nn.bias3, nn.weights4, nn.bias4, nn.weights5, nn.bias5), decimals=3)
		prediction3 = np.round(nn.forward_prop(input_image3, nn.weights1, nn.bias1, nn.weights2, nn.bias2, nn.weights3, nn.bias3, nn.weights4, nn.bias4, nn.weights5, nn.bias5), decimals=3)
		print((prediction1 + prediction2 + prediction3)/3)
		avg1 = (prediction1[0][0] + prediction2[0][0] + prediction3[0][0])/3
		avg2 = (prediction1[0][1] + prediction2[0][1] + prediction3[0][1])/3
		avg3 = (prediction1[0][2] + prediction2[0][2] + prediction3[0][2])/3
		if avg1 > avg2 and avg1 > avg3:
			oled.oled.fill(0)
			oled.oled.text("Unripe",0,1,1)
			oled.oled.show()
			print("Unripe")
			time.sleep(1)
		elif avg2 > avg1 and avg2 > avg3:
			oled.oled.fill(0)
			oled.oled.text("Ripe",0,1,1)
			oled.oled.show()
			print("Ripe")
			time.sleep(1)
		elif avg3 > avg1 and avg3 > avg2:
			oled.oled.fill(0)
			oled.oled.text("Overripe",0,1,1)
			oled.oled.show()
			print("Overripe")
			time.sleep(1)
