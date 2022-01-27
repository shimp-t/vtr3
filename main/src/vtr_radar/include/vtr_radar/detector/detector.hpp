// Copyright 2021, Autonomous Space Robotics Lab (ASRL)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * \file detector.hpp
* \brief Keypoint extraction methods for Navtech radar
 *
 * \author Keenan Burnett, Autonomous Space Robotics Lab (ASRL)
 */

#pragma once

// #include <sensor_msgs/msg/image.hpp>
// #include <vtr_lidar/pointmap/pointmap.hpp>
#include <vtr_radar/cache.hpp>


namespace vtr{
namespace radar{

using namespace vtr::lidar;


class Detector {
 public:
  Detector() {}
  virtual ~Detector() {}
  virtual void run(RadarQueryCache qdata);
};

class KStrongest : public Detector {
 public:
  KStrongest() {}
  KStrongest(int kstrong, double minr, double maxr) : kstrong_(kstrong), min_r(minr), maxr_(maxr) {}
  ~KStrongest() {}
 
 private:
  int kstrong_ = 10;
  double minr_ = 2.0;
  double maxr_ = 100.0;
};

class Cen2018 : public Detector {
 public:
  Cen2018() {}
  Cen2018(double zq, int sigma, double minr, double maxr) :
    zq_(zq), sigma_(sigma), min_r(minr), maxr_(maxr) {}
  ~Cen2018() {}
 
 private:
  double zq_ = 3.0;
  int sigma_ = 17;
  double minr_ = 2.0;
  double maxr_ = 100.0;
};

class CACFAR : public Detector {
 public:
  CACFAR() {}
  CACFAR(int width, int guard, double treshold, double minr, double maxr) :
    width_(width), guard_(guard), threshold_(treshold), min_r(minr), maxr_(maxr) {}
  ~CACFAR() {}
 
 private:
  int width_ = 40;
  int guard_ = 2;
  double threshold_ = 3.0;
  double minr_ = 2.0;
  double maxr_ = 100.0;
};

class OSCFAR : public Detector {
 public:
  OSCFAR() {}
  OSCFAR(int width, int guard, int kstat, double treshold, double minr, double maxr) :
    width_(width), guard_(guard), kstat_(kstat), threshold_(treshold), min_r(minr), maxr_(maxr) {}
  ~OSCFAR() {}
 
 private:
  int width_ = 40;
  int guard_ = 2;
  int kstat_ = 20;
  double threshold_ = 1.25;
  double minr_ = 2.0;
  double maxr_ = 100.0;
};

}  // namespace vtr
}  // namespace radar