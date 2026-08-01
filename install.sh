sudo apt install python3
sudo apt install python3-opencv
sudo apt install python3-pil
sudo apt install python3-numpy
sudo apt-get install python-smbus
sudo apt-get install i2c-tools
git clone https://github.com/adafruit/Adafruit_Python_SSD1306.git
cd Adafruit_Python_SSD1306
sudo python setup.py install
cd ..
rm -rf Adafruit_Python_SSD1306
