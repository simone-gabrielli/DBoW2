/**
* File: FSuperpoint256.cpp
* Date: March 2024
* Original Author: Dorian Galvez-Lopez
* Modified by Alejandro Fontan Villacampa for AnyFeature-VSLAM
* Description: functions for Sift128 descriptors
* License: see the LICENSE.txt file
*
*/

#include <vector>
#include <string>
#include <sstream>

#include "FClass.h"
#include "FSuperpoint256.h"

using namespace std;

namespace DBoW2 {

    // --------------------------------------------------------------------------

    void FSuperpoint256::meanValue(const std::vector<FSuperpoint256::pDescriptor>& descriptors,
        FSuperpoint256::TDescriptor& mean)
    {
        mean.resize(0);
        mean.resize(FSuperpoint256::L, 0);

        float s = descriptors.size();

        vector<FSuperpoint256::pDescriptor>::const_iterator it;
        for (it = descriptors.begin(); it != descriptors.end(); ++it)
        {
            const FSuperpoint256::TDescriptor& desc = **it;
            for (int i = 0; i < FSuperpoint256::L; i += 4)
            {
                mean[i] += desc[i] / s;
                mean[i + 1] += desc[i + 1] / s;
                mean[i + 2] += desc[i + 2] / s;
                mean[i + 3] += desc[i + 3] / s;
            }
        }
    }

    // --------------------------------------------------------------------------

    double FSuperpoint256::distance(const FSuperpoint256::TDescriptor& a, const FSuperpoint256::TDescriptor& b)
    {
        double sqd = 0.;
        for (int i = 0; i < FSuperpoint256::L; i += 4)
        {
            sqd += (a[i] - b[i]) * (a[i] - b[i]);
            sqd += (a[i + 1] - b[i + 1]) * (a[i + 1] - b[i + 1]);
            sqd += (a[i + 2] - b[i + 2]) * (a[i + 2] - b[i + 2]);
            sqd += (a[i + 3] - b[i + 3]) * (a[i + 3] - b[i + 3]);
        }
        return sqd;
    }

    // --------------------------------------------------------------------------

    std::string FSuperpoint256::toString(const FSuperpoint256::TDescriptor& a)
    {
        stringstream ss;
        for (int i = 0; i < FSuperpoint256::L; ++i)
        {
            ss << a[i] << " ";
        }
        return ss.str();
    }

    // --------------------------------------------------------------------------

    void FSuperpoint256::fromString(FSuperpoint256::TDescriptor& a, const std::string& s)
    {
        a.resize(FSuperpoint256::L);

        stringstream ss(s);
        for (int i = 0; i < FSuperpoint256::L; ++i)
        {
            ss >> a[i];
        }
    }

    // --------------------------------------------------------------------------

    void FSuperpoint256::toMat32F(const std::vector<TDescriptor>& descriptors,
        cv::Mat& mat)
    {
        if (descriptors.empty())
        {
            mat.release();
            return;
        }

        const int N = descriptors.size();
        const int L = FSuperpoint256::L;

        mat.create(N, L, CV_32F);

        for (int i = 0; i < N; ++i)
        {
            const TDescriptor& desc = descriptors[i];
            float* p = mat.ptr<float>(i);
            for (int j = 0; j < L; ++j, ++p)
            {
                *p = desc[j];
            }
        }
    }

    // --------------------------------------------------------------------------

} // namespace DBoW2
