'''
Requires the leap motion python module - will not run without it
'''
try:
    import leap

    class HandTracker():
        '''
        Higher level interface to use hand tracking with a leapmotion controller.\n
        Requires Ultraleap API https://leap2.ultraleap.com/downloads/

        ```python
        from acoustools.HandTracking import HandTracker

        tracker = HandTracker()

        with tracker.connection.open():
            tracker.start()
            running = True
            while running:
                hand = tracker.get_hands(right = False)
                if hand is not None:
                    pos = hand.palm.position
                    print(pos.x, pos.y, pos.z)
                    print(hand.grab_strength)
                    for digit in hand.digits:
                        bones = digit.bones
                        for bone in bones:
                            joint = bone.next_joint
                            print('(',joint.x, joint.y, joint.z, ')', end= ' ')
                        print()
                else:
                    print('no hand')
            ```
        '''

        def __init__(self):
            self.listener:HandListener = HandListener()
            '''
            @private
            '''


            self.connection:leap.Connection = leap.Connection()
            '''
            @private
            '''
            self.connection.add_listener(self.listener)

        
        def start(self):
            '''
            Starts the tracking
            '''
            self.connection.set_tracking_mode(leap.TrackingMode.Desktop)
        
        def get_hands(self, right:bool=True, left:bool= True):
            '''
            Returns the last scanned data
            :param right: If `True` return data on the right hand
            :param left: If `True` return data on the left hand
            '''

            if right and left:
                return self.listener.left_hand, self.listener.right_hand
            elif right:
                return self.listener.right_hand
            elif left:
                return self.listener.left_hand
            else:
                raise Exception("Need either left or right")
                        



    class HandListener(leap.Listener):
        '''
        @private
        Class to listen for hand tracking updates from a leapmotion controller.\n
        Requires Ultraleap API https://leap2.ultraleap.com/downloads/
        '''

        def __init__(self):
        
            self.left_hand = None
    
            self.right_hand = None
        

        def on_connection_event(self, event):
            '''
            prints connected when connection starts
            '''
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
                

except ImportError:
    pass