# How to create a .exe file for your code.

### Step-1 Open command prompt and install pyinstaller using following code.
    pip install pyinstaller
![alt text](image.png)

### Step-2 Update the path of the folder
use CD command to update the directory.

    cd C:\Users\akash.varasada\TOPS\TOPS_Python\DesktopApp

![alt text](image-1.png)

### Step-3 Create an .exe file.
    pyinstaller --onefile --noconsole TkinterDemo.py

![alt text](image-2.png)

### Step-4 go to the directory.dist folder should contain .exe file.
![alt text](image-3.png)
