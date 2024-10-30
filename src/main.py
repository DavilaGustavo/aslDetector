import eel
import os
import sys
import utils.state_manager
import tkinter as tk

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize eel with the correct web directory path
web_dir = os.path.join(current_dir, 'web')
eel.init(web_dir)

# Import your ASL detection modules
from utils.videoSignLanguage import videoASL
from utils.imageSignLanguage import imageASL
from utils.webcamSignLanguage import webcamASL

if __name__ == '__main__':
    try:
        # Uses Tkinter to get screen information
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()

        # Define the width and height
        window_width = screen_width - screen_width*0.35
        window_height = screen_height - screen_height*0.11

        # Calculate the central position
        pos_x = (screen_width - window_width) // 2
        pos_y = ((screen_height - window_height) // 2) - (0.02*screen_height)

        # Start the application with specific port and directory
        eel.start('index.html', 
                  port=8988,
                  mode='chrome',
                  size=(window_width, window_height),
                  position=(pos_x, pos_y))
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
        sys.exit(1)