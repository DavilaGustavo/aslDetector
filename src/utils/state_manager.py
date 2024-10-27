import eel

# Global variable to control execution
running = False

@eel.expose
def stopExecution():
    global running
    running = False

def start_execution():
    global running
    running = True

def is_running():
    global running
    return running
