cmake_minimum_required(VERSION 2.8.3)
project(mdpm_flipper)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)


catkin_package()
catkin_python_setup()


include_directories(
  ${catkin_INCLUDE_DIRS}
)
