import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/darius/turtlebot3_ws/install/Autobot_Ver1'
