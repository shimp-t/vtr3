cmake_minimum_required(VERSION 3.16)
project(zeus_description)

# Find dependencies
find_package(ament_cmake REQUIRED)

# Install launch and description files.
install(DIRECTORY
  launch urdf
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
