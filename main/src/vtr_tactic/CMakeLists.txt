cmake_minimum_required(VERSION 3.16)
project(vtr_tactic)

## Common setup for vtr packages
include("${CMAKE_CURRENT_LIST_DIR}/../vtr_common/vtr_include.cmake")


## Find dependencies
find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)

find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

find_package(lgmath REQUIRED)
find_package(steam REQUIRED)

find_package(vtr_lgmath_extensions REQUIRED)
find_package(vtr_logging REQUIRED)
find_package(vtr_mission_planning REQUIRED)
find_package(vtr_path_tracker REQUIRED)
find_package(vtr_pose_graph REQUIRED)
find_package(vtr_steam_extensions REQUIRED)


## C++ Libraries
file(GLOB_RECURSE SRC
  src/modules/*.cpp
  src/memory_manager/*.cpp
  src/cache.cpp
  src/tactic.cpp
)
add_library(${PROJECT_NAME} ${SRC})
ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  tf2 tf2_ros tf2_eigen
  sensor_msgs nav_msgs  # visualization
  lgmath steam
  vtr_logging vtr_pose_graph vtr_mission_planning vtr_path_tracker
  vtr_lgmath_extensions vtr_steam_extensions
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp
  tf2 tf2_ros tf2_eigen
  sensor_msgs nav_msgs  # visualization
  lgmath steam
  vtr_logging vtr_pose_graph vtr_mission_planning vtr_path_tracker
  vtr_lgmath_extensions vtr_steam_extensions
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS
    ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  # C++ node tests

  # Python node tests

  # Linting
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies() # Lint based on linter test_depend in package.xml
endif()

ament_package()
