# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

## Project Rubrics

## FP.1 Match 3D Objects
Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.

The code first fills the integer array to with the number of matches within each bounding box with index [prevBB][currBB] and then only keep the matches which are the highest in both Bounding Boxes


```
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
```

## FP.2 Compute Lidar-based TTC
Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame. 

The code is similar to the code in the lesson,but distances are added and the average is taken to keep the outliers from messing up the measurements

```
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
```

## FP.3 Associate Keypoint Correspondences with Bounding Boxes
Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.

The code loops over all the matches to calculate the average distance between them, and then loop over all tha matches again to only keep the keypoints which are in the Bounding Box and also less than average distance to keep the keypoint out of the bounding box which may be part of another object as well


```
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
```

## FP.4 Compute Camera-based TTC
Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.

The code computes the distance between the points in matches in a nested loop to be stored in a vector distance Ratios, which is distance ratios on keypoints matched between frames to determine the rate of scale change within an image. And in the end median of the distance Ratio is udes to measure TTC.

```
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

            // get next keypoint and its matched partner in the Curr. frame
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
```

## FP.5 Performance Evaluation 1
Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.

In the case of TTC Lidar, the values ranges from 8 sec to 16 sec which seems reasonable, still the variation is due to the outliers present since we measure the TTC using mean distances in the previous and current Frame . Other reason for this variation is due to the fact that constant velocity method which is not a perfect model. 

| Frame     | #1      | #2      | #3      | #4      | #5      | #6      | #7      | #8     | #9      | #10      | #11     | #12    | #13     | #14     | #15     | #16      | #17     | #18     |
| :------:  | :---:   | :---:   | :---:   | :---:   | :---:   | :---:   | :---:   | :---:  | :---:   | :---:    | :---:   | :---:  | :---:   | :---:   | :---:   | :---:    | :---:   | :---:   |
| TTC Lidar | 11.8809 |	13.9885 |	16.9617 |	13.4396 |	12.4126 |	14.2128 |	13.0818 |	14.4824 |	12.0435 |	12.5913 |	10.9391 |	9.9989 |	9.47086 |	9.21554 |	8.37859 |	8.88488 |	10.9544 |	8.74563 |

## FP.6 Performance Evaluation 2
Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.

Except the Harris and ORB detector, most of the detectors seemed to working fine. Harris detector detect very few keypoints which can be seen in midterm project which lead to large error in TTC estimates. Based on the result, the best estimates seems to SHITOMASI/SIFT,SHITOMASI/ORB and FAST/Brief. The result is as follows


|           |       | Frame 1   | Frame 2  | Frame 3  | Frame 4 | Frame 5 | Frame 6 | Frame 7 | Frame 8 | Frame 9 | Frame 10 | Frame 11 | Frame 12 | Frame 13 | Frame 14 | Frame 15 | Frame 16 | Frame 17 | Frame 18 |
|-----------|-------|-----------|----------|----------|---------|---------|---------|---------|---------|---------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| SHITOMASI | BRISK | 13.8948   | 12.4478  | 13.5089  | 12.9162 | 13.5185 | 13.4761 | 11.8257 | 13.1407 | 11.9106 | 14.0407  | 12.9717  | 11.8967  | 12.2852  | 10.5464  | 8.70213  | 10.8452  | 11.8197  | 8.40751  |
| SHITOMASI | BRIEF | 13.9019   | 13.0387  | 11.0044  | 14.2567 | 12.3506 | 13.5457 | 16.854  | 11.9553 | 11.9264 | 12.8285  | 11.5577  | 11.5241  | 11.4     | 11.0545  | 11.7405  | 12.2688  | 10.8657  | 8.17889  |
| SHITOMASI | ORB   | 13.6661   | 12.2204  | 11.5476  | 13.0773 | 12.1848 | 14.6025 | 12.9475 | 11.8367 | 11.8703 | 13.6717  | 11.5546  | 11.6017  | 11.6247  | 10.7101  | 10.2478  | 11.414   | 10.4945  | 8.02982  |
| SHITOMASI | FREAK | 15.3991   | 12.5392  | 11.5     | 13.4067 | 12.5078 | 13.9876 | 13.8277 | 12.1502 | 10.9782 | 13.8648  | 11.4072  | 12.7769  | 12.3243  | 11.5328  | 9.16265  | 10.9372  | 10.263   | 9.22727  |
| SHITOMASI | SIFT  | 13.826    | 12.08    | 11.4703  | 13.327  | 12.1312 | 13.2215 | 12.2347 | 12.0542 | 11.9753 | 13.368   | 11.7502  | 11.5986  | 11.6524  | 11.2222  | 9.82296  | 11.6167  | 10.9447  | 8.88054  |
| HARRIS    | BRISK | 8.81335   | 11.0081  | -0.1     | 11.3951 | 13.3698 | 12.9945 | 12.2792 | 13.5704 | nan     | -0.2     | 11.6702  | 73.7931  | 28.4161  | 56.6097  | -29.5616 | 63.3898  | 12.5848  | -0.2     |
| HARRIS    | BRIEF | 8.81335   | 11.0081  | -20.0948 | 11.3951 | 3.41473 | 13.5599 | 14.2744 | 13.5704 | 28.5939 | 10.2931  | 11.6702  | 11.3925  | 28.4161  | 56.6097  | -29.5616 | 67.1705  | 12.5848  | 12.8381  |
| HARRIS    | ORB   | 8.81335   | 10.586   | nan      | 12.4858 | 37.3809 | 12.9945 | 13.497  | 12.6385 | 28.5939 | 10.2931  | 11.6702  | 11.1055  | 13.4327  | 56.6097  | -25.2781 | 67.1705  | 12.5848  | 0.1      |
| HARRIS    | FREAK | 8.69538   | 10.586   | nan      | 11.6519 | 13.3698 | 13.5599 | 12.3379 | 12.6385 | nan     | 10.2931  | 11.2142  | 11.1055  | 13.301   | -20.1621 | -0.2     | 63.3929  | nan      | -inf     |
| HARRIS    | SIFT  | 8.81335   | 18.2178  | -20.0948 | 11.3794 | 37.3809 | 13.5599 | 13.497  | 13.5704 | 28.5939 | 10.2931  | 11.6702  | -inf     | 13.3551  | 56.6097  | -inf     | 66.0338  | 11.7964  | -inf     |
| FAST      | BRISK | 12.3      | 13.9829  | 15.6402  | 14.0808 | 27.3785 | 14.2302 | 17.5414 | 12.0122 | 13.0566 | 12.3835  | 13.5948  | 12.7922  | 11.868   | 12.2376  | 11.8818  | 12.433   | 10.3629  | 12.2338  |
| FAST      | BRIEF | 10.3507   | 12.5159  | 11.3018  | 14.0692 | 13.3471 | 13.1753 | 11.6025 | 11.4596 | 11.7261 | 13.9083  | 13.3673  | 10.8684  | 12.0729  | 10.4235  | 10.7673  | 11.5631  | 8.23412  | 12.5199  |
| FAST      | ORB   | 11.4035   | 12.1233  | 21.6278  | 12.3708 | 9.96634 | 13.422  | 12.4571 | 11.8227 | 11.8135 | 12.6222  | 13.1149  | 13.0459  | 12.5118  | 11.9791  | 11.1063  | 12.2613  | 9.31158  | 12.2068  |
| FAST      | FREAK | 34.9752   | -80.7776 | 16.7839  | 13.9603 | 11.4421 | 19.3412 | 19.2702 | 12.6222 | 11.9735 | 14.0253  | -inf     | 20.2683  | inf      | 12.3866  | 12.3572  | 12.4653  | inf      | 12.8877  |
| FAST      | SIFT  | 11.199    | 12.2335  | 14.1981  | 13.794  | 21.3581 | 13.1897 | 13.327  | 11.8507 | 12.9544 | 13.8308  | 13.7878  | 11.8507  | 12.2419  | 11.3881  | 10.8729  | 12.24    | 8.3405   | 11.6025  |
| BRISK     | BRISK | 12.5978   | 20.5746  | 13.3577  | 17.7678 | 33.4482 | 16.2737 | 15.486  | 16.1664 | 14.9952 | 11.3675  | 12.8434  | 11.1817  | 11.8651  | 11.6739  | 11.2441  | 11.2436  | 9.53204  | 11.7408  |
| BRISK     | BRIEF | 13.2855   | 26.8904  | 12.7004  | 18.2695 | 15.2309 | 20.9227 | 14.9545 | 18.9793 | 13.132  | 10.4428  | 14.2903  | 12.4172  | 10.8361  | 11.9398  | 10.4242  | 13.1142  | 9.26558  | 11.9201  |
| BRISK     | ORB   | 12.8852   | 16.2077  | 13.9095  | 15.6583 | 18.5859 | 17.9553 | 15.986  | 16.0324 | 13.7378 | 11.3782  | 12.7217  | 10.8713  | 11.1476  | 11.4329  | 12.1457  | 11.1185  | 10.0809  | 11.6644  |
| BRISK     | FREAK | 15.7781   | 18.596   | 12.7562  | 14.3906 | 21.2706 | 14.4235 | 15.4536 | 17.8591 | 16.4994 | 13.4361  | 14.1501  | 12.4027  | 12.5763  | 11.9776  | 12.5515  | 10.4555  | 8.67017  | 10.8595  |
| BRISK     | SIFT  | 12.9834   | 14.3563  | 15.6621  | 10.5846 | 27.8489 | 15.1563 | 14.1326 | 18.252  | 17.2713 | 14.0544  | 13.638   | 12.7026  | 11.64    | 10.7575  | 12.6832  | 11.4182  | 11.0563  | 14.2147  |
| ORB       | BRISK | 13.9541   | 15.6278  | 11.669   | 4.2674  | 5.38829 | 9.24619 | -inf    | 10.9114 | 11.3831 | 11.3956  | 73.7615  | -inf     | 12.8     | 78.3786  | 19.071   | 12.6955  | 12.4968  | 27.5113  |
| ORB       | BRIEF | 13.0283   | 22.6412  | 2.4554   | 20.3785 | 17.7994 | -inf    | 2.2677  | 19.5822 | 24.8992 | 11.0543  | 1.0154   | 21.5419  | 13.7328  | 9.54006  | 13.7016  | 14.5151  | 14.1486  | 17.8921  |
| ORB       | ORB   | 14.3109   | 9.98567  | 12.9923  | 10.1864 | 21.7176 | 12.7341 | 17.4772 | 9.02344 | -inf    | 11.3956  | 77.8781  | 35.8486  | 10.114   | 13.7685  | 13.594   | 10.0471  | 18.6979  | 2.5339   |
| ORB       | FREAK | 12.1791   | 38.7882  | 11.1077  | 11.0409 | 7.4482  | 10.9269 | -inf    | 12.7778 | 13.0837 | -inf     | 7.90955  | 2.4537   | 6.93371  | 35.3185  | 8.68802  | 7.46502  | 15.0271  | 7.57184  |
| ORB       | SIFT  | -0.397746 | 13.599   | 17.4309  | -inf    | -3.3577 | -inf    | -inf    | 10.0577 | 13.1285 | 11.1389  | 9.20223  | -inf     | -13.3757 | 53.0034  | 23.5777  | 9.48983  | 17.4675  | 24.3118  |
| AKAZE     | BRISK | 12.2899   | 13.4781  | 13.0026  | 14.2488 | 13.9519 | 14.3014 | 15.8042 | 13.8407 | 13.5036 | 11.8209  | 12.443   | 10.0243  | 10.3434  | 9.85734  | 9.81599  | 10.1316  | 9.34993  | 8.71812  |
| AKAZE     | BRIEF | 12.9808   | 14.5297  | 12.7394  | 14.2197 | 15.0016 | 12.9594 | 15.9106 | 14.0791 | 13.7728 | 11.9479  | 12.5053  | 11.3185  | 10.0991  | 10.0535  | 9.70276  | 9.72802  | 9.17787  | 9.04661  |
| AKAZE     | ORB   | 12.6985   | 14.3581  | 12.6035  | 14.3523 | 15.4132 | 14.1421 | 16.0305 | 13.6042 | 12.8314 | 11.76    | 12.2094  | 11.9534  | 10.4345  | 10.3685  | 10.2267  | 9.71731  | 8.97363  | 8.59537  |
| AKAZE     | FREAK | 12.033    | 13.6591  | 12.9431  | 14.3594 | 15.0244 | 14.2195 | 14.92   | 14.1319 | 13.2972 | 11.5913  | 11.9667  | 10.3178  | 11.0849  | 9.51794  | 9.79527  | 9.96094  | 8.96422  | 8.41151  |
| AKAZE     | AKAZE | 12.4678   | 13.7432  | 12.8172  | 13.8582 | 15.3012 | 13.6559 | 15.6719 | 14.0272 | 13.5902 | 11.4403  | 12.1519  | 11.2212  | 11.0477  | 1.0604   | 10.3881  | 9.96215  | 9.19227  | 8.79442  |
| AKAZE     | SIFT  | 12.854    | 13.942   | 12.8667  | 14.0071 | 15.0016 | 13.6537 | 15.4734 | 13.4917 | 13.7655 | 11.5597  | 12.0663  | 10.847   | 11.083   | 10.4941  | 10.0915  | 10.0977  | 9.35097  | 8.91654  |
| SIFT      | BRISK | 11.5285   | 13.5976  | 13.2768  | 22.334  | 15.1253 | 11.0631 | 13.782  | 14.0906 | 12.8436 | 10.838   | 11.2389  | 10.6742  | 10.0387  | 9.84427  | 9.96945  | 8.95836  | 8.78839  | 8.62013  |
| SIFT      | BRIEF | 12.0691   | 13.3859  | 13.9367  | 21.1887 | 13.6163 | 11.9135 | 14.1552 | 16.1864 | 13.0849 | 10.4059  | 11.4159  | 11.2921  | 9.98918  | 10.0373  | 9.2644   | 8.95836  | 8.69186  | 8.81472  |
| SIFT      | FREAK | 12.9077   | 14.1018  | 14.688   | 29.2893 | 14.1656 | 12.5276 | 16.3972 | 15.8044 | 28.2934 | 11.8289  | 11.8019  | 11.842   | 9.16488  | 9.39099  | 9.07095  | 10.4146  | 8.61237  | 8.82325  |
| SIFT      | SIFT  | 12.2705   | 12.5959  | 13.833   | 21.8392 | 14.061  | 12.7921 | 15.708  | 18.7953 | 19.1447 | 10.9943  | 12.3604  | 11.4586  | 10.9039  | 10.6575  | 10.5037  | 9.29747  | 8.70533  | 8.77684  |






