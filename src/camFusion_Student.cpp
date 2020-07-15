
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    //Loop to find the average distance between kpt Matches
    float Dist=0.0;
    for (cv::DMatch match : kptMatches)
    {
        Dist=Dist+match.distance;
    }
    float dist_average=Dist/kptMatches.size();
    
    // Loop over all matches in the current frame
    for (cv::DMatch match : kptMatches)
    {
        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt))
        {
            if (match.distance<dist_average)
            {
                boundingBox.kptMatches.push_back(match);
            }
            
        }
    }
}


void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }// eof inner loop over all matched kpts
    }// eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    /*
    // compute camera-based TTC from distance ratios mean value
    std::sort(distRatios.begin(), distRatios.end());
    double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();

    double dT = 1 / frameRate;
    TTC = -dT / (1 - meanDistRatio);
    */
    // If median would be used
    
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex + 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex+1]; // compute median dist. ratio to remove outlier influence

    TTC = (-1.0 / frameRate) / (1 - medDistRatio);    
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
     // auxiliary variables
    double dT = 1/frameRate;        // time between two measurements in seconds
    double laneWidth = 4.0; // assumed width of the ego lane

    // find closest distance to Lidar points within ego lane
    double Dist_Prev, Dist_Curr;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        if (abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            Dist_Prev += it->x;
        }
    }
    
    Dist_Prev=Dist_Prev/lidarPointsPrev.size(); //average distances in x direction

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
       if (abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            Dist_Curr += it->x; ////average distances in x direction
        }
    }

    Dist_Curr=Dist_Curr/lidarPointsCurr.size();

    // compute TTC from both measurements
    TTC = Dist_Curr * dT / (Dist_Prev - Dist_Curr);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
     
    // DMatch contains two keypoint indices, queryIdx and trainIdx, based on the order of image arguments to match.
    // https://docs.opencv.org/4.1.0/db/d39/classcv_1_1DescriptorMatcher.html#a0f046f47b68ec7074391e1e85c750cba

    //list size of boundingboxes in both frames which ranges from 6 to 11 Bpunding Box in each frame (image)
    int p = prevFrame.boundingBoxes.size(); 
    int c = currFrame.boundingBoxes.size();
    
    //variable to count keypoints
    int pt_number[p][c]={};

    //loop over matches
    for (auto it = matches.begin(); it != matches.end() - 1; it++) // Matches are in the range of 1500-1800 depending on detector and descriptor combination
    {
        cv::KeyPoint pkp = prevFrame.keypoints[it->queryIdx]; //previous keypoint
        bool pkp_found = false;

        cv::KeyPoint ckp = currFrame.keypoints[it->trainIdx]; // current keypoint
        bool ckp_found = false;

        std::vector<int> pkp_id, ckp_id; // vector to store ids of the keypoints which are in the bounding boxes

        //looping over all previous frame bounding boxes to check if they have previous keypoints
        for (int i = 0; i < p; i++)
        {
            if (prevFrame.boundingBoxes[i].roi.contains(pkp.pt))  //if found than push it to previous keypoint id          
             {
                pkp_found = true;
                pkp_id.push_back(i);
             }
        }
        ////looping over all current frame bounding boxes to check if they have current keypoints
        for (int i = 0; i < c; i++) 
        {
            if (currFrame.boundingBoxes[i].roi.contains(ckp.pt))   // same as previous frame case        
            {
                ckp_found= true;
                ckp_id.push_back(i);
            }
        }
        // if kps are found in both data frames
        if (pkp_found && ckp_found) 
        {
            for (auto id_prev: pkp_id) // iterating through previous keypoint ids present in prevframe Roi
            {
                for (auto id_curr: ckp_id) //iterating through current keypoint ids present in currFrame Roi
                {
                    pt_number[id_prev][id_curr] += 1; // increase the integer +1
                }
            }
        }
    }
   
    // Now to find best matches 
    //Looping through prevFrame BoundingBoxes size
    for (int i = 0; i < p; i++) 
    {  
         int max_count = 0;
         int id_max = 0;
        //Looping through currFrame BoundingBoxes size 
        for (int j = 0; j < c; j++)
        {
            if (pt_number[i][j] > max_count)
            {  
                max_count = pt_number[i][j];
                id_max = j;
            }
        }
        bbBestMatches[i] = id_max;
    }
}
