sudo apt install python3
sudo apt install python3-pip
sudo apt install python3-opencv
sudo apt install fswebcam
sudo apt install python3-pil
sudo apt install python3-numpy
sudo apt-get install python3-smbus
sudo apt-get install i2c-tools
sudo apt install python3-picamera2

pip3 install cupy --break-system-packages
pip3 install board --break-system-packages
pip3 install keyboard --break-system-packages
pip3 install adafruit-circuitpython-ssd1306 adafruit-blinka --break-system-packages

echo "Enter username to automatically log in with"
read username

autologstr = '[Service]
ExecStart=
ExecStart=-/sbin/agetty --noclear --autologin ' 
autologstr += username
autologstr += ' %I $TERM'
sudo echo autologstr  > /etc/systemd/system/getty@tty1.service.d/autologin.conf
