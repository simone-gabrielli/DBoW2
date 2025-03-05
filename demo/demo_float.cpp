/**
 * File: Demo_256D.cpp
 * Description: DBoW2 demo adapted for 256D float descriptors
 */

#include <iostream>
#include <vector>
#include <DBoW2.h>
#include <opencv2/opencv.hpp>

using namespace DBoW2;
using namespace std;

// Number of training images
const int NIMAGES = 4;
const int DESC_DIM = 256;  // 256D Float descriptors

void loadFeatures(vector<vector<vector<float>>>& features);
void changeStructure(const cv::Mat& plain, vector<vector<float>>& out);
void testVocCreation(const vector<vector<vector<float>>>& features);
void testDatabase(const vector<vector<vector<float>>>& features);

void wait()
{
    cout << endl << "Press enter to continue" << endl;
    getchar();
}

int main()
{
    vector<vector<vector<float>>> features;
    loadFeatures(features);

    testVocCreation(features);
    wait();
    testDatabase(features);

    return 0;
}

// ----------------------------------------------------------------------------
// Load 256D Float Descriptors (Simulated SuperPoint)
void loadFeatures(vector<vector<vector<float>>>& features)
{
    features.clear();
    features.reserve(NIMAGES);

    cout << "Extracting 256D float descriptors..." << endl;
    for (int i = 0; i < NIMAGES; ++i)
    {
        stringstream ss;
        ss << "images/image" << i << ".png";

        cv::Mat image = cv::imread(ss.str(), cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            cerr << "Error: Could not load image " << ss.str() << endl;
            continue;
        }

        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        // Simulating 256D Float Descriptors (Replace with real feature extractor)
        int num_keypoints = 500;
        descriptors = cv::Mat(num_keypoints, DESC_DIM, CV_32F);
        cv::randu(descriptors, 0, 1);  // Random values in [0,1]

        features.push_back(vector<vector<float>>());
        changeStructure(descriptors, features.back());
    }
}

// ----------------------------------------------------------------------------
// Convert OpenCV Mat to std::vector<float>
void changeStructure(const cv::Mat& plain, vector<vector<float>>& out)
{
    out.resize(plain.rows);
    for (int i = 0; i < plain.rows; ++i)
    {
        out[i].resize(plain.cols);
        memcpy(out[i].data(), plain.ptr<float>(i), plain.cols * sizeof(float));
    }
}

// ----------------------------------------------------------------------------
// Create a Vocabulary for 256D Float Descriptors
void testVocCreation(const vector<vector<vector<float>>>& features)
{
    const int k = 10;  // Branching factor
    const int L = 3;   // Depth levels
    const WeightingType weight = TF_IDF;
    const ScoringType scoring = L2_NORM;  // Use L2 distance instead of Hamming

    Superpoint256Vocabulary voc(k, L, weight, scoring);

    cout << "Creating a " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;

    cout << "Vocabulary information: " << endl << voc << endl;

    // Test vocabulary
    cout << "Matching images against themselves (0 low, 1 high): " << endl;
    BowVector v1, v2;
    for (int i = 0; i < NIMAGES; i++)
    {
        voc.transform(features[i], v1);
        for (int j = 0; j < NIMAGES; j++)
        {
            voc.transform(features[j], v2);
            double score = voc.score(v1, v2);
            cout << "Image " << i << " vs Image " << j << ": " << score << endl;
        }
    }

    // Save vocabulary
    cout << endl << "Saving vocabulary..." << endl;
    voc.save("float_vocabulary.yml.gz");
    cout << "Done" << endl;
}

// ----------------------------------------------------------------------------
// Create and Query a Database for 256D Float Descriptors
void testDatabase(const vector<vector<vector<float>>>& features)
{
    cout << "Creating a small database..." << endl;

    // Load vocabulary
    Superpoint256Vocabulary voc("float_vocabulary.yml.gz");
    Superpoint256Database db(voc, false, 0);

    // Add images to database
    for (int i = 0; i < NIMAGES; i++)
    {
        db.add(features[i]);
    }

    cout << "... done!" << endl;
    cout << "Database information: " << endl << db << endl;

    // Query the database
    cout << "Querying the database: " << endl;
    QueryResults ret;
    for (int i = 0; i < NIMAGES; i++)
    {
        db.query(features[i], ret, 4);
        cout << "Searching for Image " << i << ". " << ret << endl;
    }

    cout << endl;

    // Save database
    cout << "Saving database..." << endl;
    db.save("float_db.yml.gz");
    cout << "... done!" << endl;

    // Load database again
    cout << "Retrieving database once again..." << endl;
    Superpoint256Database db2("float_db.yml.gz");
    cout << "... done! This is: " << endl << db2 << endl;
}
