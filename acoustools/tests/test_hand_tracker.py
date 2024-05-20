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

        input()