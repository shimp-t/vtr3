cmake_minimum_required(VERSION 3.16)
project(vtr_vision)

# Common setup for vtr packages
include("${CMAKE_CURRENT_LIST_DIR}/../vtr_common/vtr_include.cmake")

# Find dependencies
find_package(ament_cmake REQUIRED)

find_package(OpenCV REQUIRED)

find_package(gpusurf REQUIRED) # currently assuming this is required
find_package(lgmath REQUIRED)
find_package(steam REQUIRED)

find_package(vtr_common REQUIRED)
find_package(vtr_logging REQUIRED)
find_package(vtr_messages REQUIRED)
find_package(vtr_storage REQUIRED)
find_package(vtr_tactic REQUIRED)

file(GLOB_RECURSE SRC src/*.cpp)
add_library(${PROJECT_NAME} ${SRC})
ament_target_dependencies(${PROJECT_NAME}
  OpenCV
  gpusurf lgmath steam
  vtr_common vtr_logging vtr_messages vtr_storage vtr_tactic
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  gpusurf lgmath steam
  vtr_common vtr_logging vtr_messages vtr_storage vtr_tactic
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
  # ament_add_gtest(augmenter_tests test/features/augmenter_tests.cpp)
  # target_link_libraries(augmenter_tests ${PROJECT_NAME})
  ament_add_gtest(bow_tests test/features/bow_tests.cpp)
  target_link_libraries(bow_tests ${PROJECT_NAME})
  ament_add_gtest(ransac_dataset_test test/integration/ransac_dataset_test.cpp)
  target_link_libraries(ransac_dataset_test ${PROJECT_NAME})
  ament_add_gtest(conversion_test test/messages/conversion_test.cpp)
  target_link_libraries(conversion_test ${PROJECT_NAME})
  ament_add_gtest(sampler_tests test/outliers/sampler_tests.cpp)
  target_link_libraries(sampler_tests ${PROJECT_NAME})
  ament_add_gtest(stereo_camera_model_tests test/outliers/stereo_camera_model_tests.cpp)
  target_link_libraries(stereo_camera_model_tests ${PROJECT_NAME})

  # Copy sample dataset from src to build
  file(COPY samples/data/ DESTINATION sample_data)

  # Linting
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies() # Lint based on linter test_depend in package.xml

  # Other
  add_executable(ransac_demo samples/ransac_demo.cpp)
  target_link_libraries(ransac_demo ${PROJECT_NAME})
endif()

ament_package()

