import numpy as np
import json
import imgtoarray as ita
import trainer as tr
import asdlkfj as deez
import camera as c
import neural_network as nn
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

training_data_paths = ["training_images/unripe", "training_images/ripe", "training_images/overripe"]


print("Type \"help\" for a list of commands")

#will implement preference data harvested from summer camp

while True:
    command = input("> ")
    if command == "help":
        print("datamanage, train, dump, banana, test")
    elif command == "datamanage":
        ita.name_training_data(training_data_paths[0])
        ita.name_training_data(training_data_paths[1])
        ita.name_training_data(training_data_paths[2])
        #ita.name_training_data(training_data_paths[3])
    elif command == "train":
        input_faces = ita.read_training_data_rgb(training_data_paths)
        with open("training_data/desired_training_outputs.txt", "r") as file:
            desired_training_outputs = json.load(file)
        tr.train(input_faces, np.array(desired_training_outputs))
    elif command == "dump":
        tr.dump_corrections(training_data_paths)
        with open("training_data/desired_training_outputs.txt", "r") as file:
            desired_training_outputs = json.load(file)
    elif command == "banana":
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
        while True:
            c.main()
            input_image1 = ita.convert_rgb(f"camera/banana1.png")
            input_image2 = ita.convert_rgb(f"camera/banana2.png")
            input_image3 = ita.convert_rgb(f"camera/banana3.png")
            prediction1 = np.round(nn.forward_prop(input_image1, nn.weights1, nn.bias1, nn.weights2, nn.bias2, nn.weights3, nn.bias3, nn.weights4, nn.bias4, nn.weights5, nn.bias5), decimals=3)
            prediction2 = np.round(nn.forward_prop(input_image2, nn.weights1, nn.bias1, nn.weights2, nn.bias2, nn.weights3, nn.bias3, nn.weights4, nn.bias4, nn.weights5, nn.bias5), decimals=3)
            prediction3 = np.round(nn.forward_prop(input_image3, nn.weights1, nn.bias1, nn.weights2, nn.bias2, nn.weights3, nn.bias3, nn.weights4, nn.bias4, nn.weights5, nn.bias5), decimals=3)
            print((prediction1 + prediction2 + prediction3)/3)
            avg1 = (prediction1[0][0] + prediction2[0][0] + prediction3[0][0])/3
            avg2 = (prediction1[0][1] + prediction2[0][1] + prediction3[0][1])/3
            avg3 = (prediction1[0][2] + prediction2[0][2] + prediction3[0][2])/3
            if  avg1 > avg2 and avg1 > avg3:
                print("Underripe")
            elif avg2 > avg1 and avg2 > avg3:
                print("Ripe")
            elif avg3 > avg1 and avg3 > avg2:
                print("Overripe")
                
    elif command == "test":
        import test
                
    else:
        print("Invalid command")
        
    
    
