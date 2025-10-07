cmake_minimum_required(VERSION 3.0.2)
project(egg)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

find_package(catkin REQUIRED COMPONENTS roscpp rospy sensor_msgs std_msgs)

include_directories(${catkin_INCLUDE_DIRS})

catkin_python_setup()
catkin_package()
