import numpy as np
import json
import imgtoarray as ita
import asdlkfj as deez
import camera as c
import oled
import trainer as tr

print(deez.nuts())

def dingus(input_layer, weights1, bias1, weights2, bias2, weights3, bias3, weights4, bias4, weights5, bias5):
    l1 = np.dot(input_layer, weights1) + bias1
    l1_outputs = tr.relu(l1)
    l2 = np.dot(l1_outputs, weights2) + bias2
    l2_outputs = tr.relu(l2)
    l3 = np.dot(l2_outputs, weights3) + bias3
    l3_outputs = tr.relu(l3)
    l4 = np.dot(l3_outputs, weights4) + bias4
    l4_outputs = tr.relu(l4)
    l5 = np.dot(l4_outputs, weights5) + bias5
    l5_outputs = tr.softmax(l5)
    return l5_outputs

elements = 144
desired_training_outputs = [[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],
                            [0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],
                            [0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],
                            [1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],
                            [0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],
                            [0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1],
                            ]



oled.test()
while True:
    command = "banana"
    if command == "banana":
        with open("training_data/weights1.txt", "r") as file:
            weights1 = json.load(file)
        with open("training_data/bias1.txt", "r") as file:
            bias1 = json.load(file)
        with open("training_data/weights2.txt", "r") as file:
            weights2 = json.load(file)
        with open("training_data/bias2.txt", "r") as file:
            bias2 = json.load(file)
        with open("training_data/weights3.txt", "r") as file:
            weights3 = json.load(file)
        with open("training_data/bias3.txt", "r") as file:
            bias3 = json.load(file)
        with open("training_data/weights4.txt", "r") as file:
            weights4 = json.load(file)
        with open("training_data/bias4.txt", "r") as file:
            bias4 = json.load(file)
        with open("training_data/weights5.txt", "r") as file:
            weights5 = json.load(file)
        with open("training_data/bias5.txt", "r") as file:
            bias5 = json.load(file)
        #path = input("Enter image filename: ")
        while True:
            c.main()
            input_image1 = ita.convert(f"images/banana1.png")
            input_image2 = ita.convert(f"images/banana2.png")
            input_image3 = ita.convert(f"images/banana3.png")
            prediction1 = np.round(dingus(input_image1, weights1, bias1, weights2, bias2, weights3, bias3, weights4, bias4, weights5, bias5), decimals=3)
            prediction2 = np.round(dingus(input_image2, weights1, bias1, weights2, bias2, weights3, bias3, weights4, bias4, weights5, bias5), decimals=3)
            prediction3 = np.round(dingus(input_image3, weights1, bias1, weights2, bias2, weights3, bias3, weights4, bias4, weights5, bias5), decimals=3)
            print((prediction1 + prediction2 + prediction3)/3)
            if (prediction1[0][1] + prediction2[0][1] + prediction3[0][1])/3 >= .5:
                oled.oled.fill(0)
                oled.oled.text("Mediocre",0,1,1)
                oled.oled.show()
                print("Mid")
            elif (prediction1[0][0] + prediction2[0][0] + prediction3[0][0])/3 >= .5:
                oled.oled.fill(0)
                oled.oled.text("Enjoyable",0,1,1)
                oled.oled.show()
                print("Good")
            elif (prediction1[0][2] + prediction2[0][2] + prediction3[0][2])/3 >= .5:
                oled.oled.fill(0)
                oled.oled.text("Poor",0,1,1)
                oled.oled.show()
                print("Bad")
        
    
    
