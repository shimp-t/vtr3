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
 * \file detector.cpp
 * \brief Keypoint extraction methods for Navtech radar
 *
 * \author Keenan Burnett, Autonomous Space Robotics Lab (ASRL)
 */
#include <vtr_radar/detector/detector.hpp>


namespace vtr {
namespace radar {

using namespace std;
using namespace vtr::lidar;

static bool sortbysec(const pair<int,double> &a,
    const pair<int,double> &b) {
    return (a.second > b.second);
}

KStrongest::run(tactic::QueryCache &qdata0) {
    auto &qdata = dynamic_cast<RadarQueryCache &>(qdata0);
    // Input
    const auto &raw_scan = *qdata.raw_scan.fallback();
    const auto &res = *qdata.radar_resolution.fallback();
    const auto &azimuth_times = *qdata.azimuth_times.fallback();
    const auto &azimuth_angles = *qdata.azimuth_angles.fallback();
    // Output
    auto &raw_pointcloud_pol = *qdata.raw_pointcloud_pol.fallback();
    auto &raw_pointcloud_times = *qdata.raw_pointcloud_times.fallback();

    const uint rows = raw_scan.rows;
    const uint cols = raw_scan.cols;
    uint mincol = minr_ * res;
    if (mincol > cols || mincol < 0) mincol = 0;
    uint maxcol = maxr_ * res;
    if (maxcol > cols || maxcol < 0) maxcol = cols;
    const uint N = maxcol - mincol;
    
    raw_pointcloud_pol.clear();
    raw_pointcloud_times.clear();

#pragma omp parallel for 
    for (uint i = 0; i < rows; ++i) {
        vector<pair<int, double>> intens(N, 0);
        double mean = 0;
        for (uint j = mincol; j < maxcol; ++j) {
            intens[j - mincol] = make_pair(j, raw_scan.at(i, j));
            mean += raw_scan.at(i, j);
        }
        mean /= N;

        // sort intensities in descending order
        sort(intens.begin(), intens.end(), sortbysec);
        const double azimuth = azimuth_angles[i];
        const double time = azimuth_times[i];
        const double thres = mean * threshold_;
        std::vector<PointXYZ> polar;
        for (int j = 0; j < kstrong_; ++j) {
            if (intens[j].second < thres)
                break;
            const uint rad = intens[j].first * res;
            polar.push_back(rad, 0, azimuth);
        }
        const std::vector<double> times(polar.size(), time);
        #pragma omp critical
        {
        raw_pointcloud_pol.insert(raw_pointcloud_pol.end(), polar.begin(), polar.end());
        raw_pointcloud_times.insert(raw_pointcloud_times.end(), times.begin(), times.end());
        }
    }
}

Cen2018::run(tactic::QueryCache &qdata0) { }  // TODO

CACFAR::run(tactic::QueryCache &qdata0) { }  // TODO

OSCFAR::run(tactic::QueryCache &qdata0) { }  // TODO

}  // namespace radar
}  // namespace vtr