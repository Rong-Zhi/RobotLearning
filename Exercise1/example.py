from jointCtlComp import *
from taskCtlComp import *


# Controller in the joint space. The robot has to reach a fixed position.

#Control_flag = ['P','PID','PD_Grav','ModelBased']
for flag in Control_flag:
    jointCtlComp([flag] , 1, 0 , 'MyJoint')

# Same controller, but this time the robot has to follow a fixed trajectory.
for flag in Control_flag:
    jointCtlComp([flag] , 0, 0 , 'MyJoint')

# Controller in the task space.
#taskCtlComp(['JacNullSpace'], 10, 'exampleJoint')
taskCtlComp(['JacNullSpace'], 0.1, 'exampleJoint')

#raw_input('Finish?')
