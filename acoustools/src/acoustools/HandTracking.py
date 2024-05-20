import leap

class HandListener(leap.Listener):

    def __init__(self):
        self.left_hand = None
        self.right_hand = None

    def on_connection_event(self, event):
        print("Connected")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()

        print(f"Found device {info.serial}")

    def on_tracking_event(self, event, reset=True):
        if reset:
            self.left_hand = None
            self.right_hand = None
        
        for hand in event.hands:
            if str(hand.type) == "HandType.Left":
                self.left_hand = hand
            else:
                self.right_hand = hand
            

class HandTracker():

    def __init__(self):
        self.listener = HandListener()

        self.connection = leap.Connection()
        self.connection.add_listener(self.listener)

    
    def start(self):
        self.connection.set_tracking_mode(leap.TrackingMode.Desktop)
    
    def get_hands(self, right=True, left= True):

        if right and left:
            return self.listener.left_hand, self.listener.right_hand
        elif right:
            return self.listener.right_hand
        elif left:
            return self.listener.left_hand
        else:
            raise Exception("Need either left or right")
                    


