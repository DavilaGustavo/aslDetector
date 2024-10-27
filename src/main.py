import eel
import os
import sys
import state_manager

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize eel with the correct web directory path
web_dir = os.path.join(current_dir, 'web')
eel.init(web_dir)

# Import your ASL detection modules
from executionCode.videoSignLanguage import videoASL
from executionCode.webcamSignLanguage import webcamASL

if __name__ == '__main__':
    try:
        # Start the application with specific port and directory
        eel.start('index.html', 
                  port=8988,
                  mode='chrome',
                  size=(1000, 800))
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
        sys.exit(1)