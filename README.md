# Bananai - Project status
- Able to recognize 4x4 smiley faces and sad faces
- WE DO BANANAS NOW (poorly - 4 hidden layers)
- Added camera support
- Added rpi3 with 128x64 oled support

# Bananai - Installation
- PC
  - Run main.py

- RPI
  - Install Raspberry Pi OS Lite
  - Run "sudo raspi-config" and enable ARM64 I2C
  - Run install.sh to install dependencies and enable automatic login
  - edit your .bashrc to include "cd Bananai/rpi" and "sudo -E python3 main.py"
  - Reboot your system
