
#include <stdio.h>

#include <iostream>
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/calib3d/calib3d_c.h"

#include "opencv2/reg/mapaffine.hpp"
#include "opencv2/reg/mappergradaffine.hpp"
#include "opencv2/reg/mappergradshift.hpp"
#include "opencv2/reg/mappergradproj.hpp"
#include "opencv2/reg/mapshift.hpp"
#include "opencv2/reg/mapprojec.hpp"
#include "opencv2/reg/mapperpyramid.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::reg;
void readme();
static void showDifference(const Mat& image1, const Mat& image2, const char* title);
static void testAffine(const Mat& img1,const Mat& img2);
bool flag = false;
static const char* DIFF_IM = "Image difference";
static const char* DIFF_REGPIX_IM = "Image difference: pixel registered";
/*
 * @function main
 * @brief Main function
 */
 int main( int argc, char** argv )
{

  if( argc != 3 )
  { readme(); return -1; }
  Mat img_1 = imread( argv[1], IMREAD_GRAYSCALE );
  Mat img_2 = imread( argv[2], IMREAD_GRAYSCALE );
  if( !img_1.data || !img_2.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }


  //-- Step 1: Detect the keypoints using SIRF Detector, compute the descriptors
  //int minHessian = 400;
  Ptr<Feature2D> detector = SIFT::create();
 // detector->setHessianThreshold(minHessian);
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;
  detector->detectAndCompute( img_1, Mat(), keypoints_1, descriptors_1 );
  detector->detectAndCompute( img_2, Mat(), keypoints_2, descriptors_2 );

  //-- Step 2: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_1, descriptors_2, matches );
  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_1.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );



  std::vector< DMatch > good_matches;
  for( int i = 0; i < descriptors_1.rows; i++ )
  { if( matches[i].distance < max(2*min_dist, 0.02) )
    { good_matches.push_back( matches[i]); }
  }

  //-- Draw only "good" matches
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


  //-- Show detected matches
  imshow( "Good Matches", img_matches );
    for( int i = 0; i < (int)good_matches.size(); i++ )
    {
      printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx );
    }

   	string imgMatches = "matches.jpg";
    imwrite(imgMatches, img_matches);


    std::vector< Point2f > obj;
	std::vector< Point2f > scene;

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_2[good_matches[i].trainIdx].pt);
	}


    Mat result;
	// Find the Homography Matrix
	Mat H = findHomography(obj, scene, CV_RANSAC);
	// Use the Homography Matrix to warp the images

	Mat Hinv = H.inv();
	warpPerspective(img_2, result, Hinv, img_1.size());

	cout << "--- Feature method\n" << H << endl;

	Mat imf1, resf;
	img_1.convertTo(imf1, CV_64FC3);
	result.convertTo(resf, CV_64FC3);

	testAffine(imf1,resf);

  waitKey(0);
  return 0;

}

static void showDifference(const Mat& image1, const Mat& image2, const char* title)
{
	Mat img1, img2;
	image1.convertTo(img1, CV_32FC3);
	image2.convertTo(img2, CV_32FC3);


	if (img1.channels() != 1)
		cvtColor(img1, img1, CV_RGB2GRAY);
	if (img2.channels() != 1)
		cvtColor(img2, img2, CV_RGB2GRAY);

	Mat imgDiff;
	img1.copyTo(imgDiff);


	imgDiff -= img2;
	imgDiff /= 2.f;
	imgDiff += 128.f;

	Mat imgSh;
	imgDiff.convertTo(imgSh, CV_8UC3);
	imshow(title, imgSh);


	  string fileName = "warpedImage.jpg";
            //save the warped image
     imwrite(fileName, imgSh);
	}




static void testAffine(const Mat& img1,const Mat& img2)
{

	// Warp original image
	Matx<double, 2, 2> linTr(1., 0.1, -0.01, 1.);
	Vec<double, 2> shift(1., 1.);
	MapAffine mapTest(linTr, shift);
	mapTest.warp(img1, img2);
	//showDifference(img1, img2, DIFF_IM);

	// Register
	Ptr<MapperGradAffine> mapper = makePtr<MapperGradAffine>();
	MapperPyramid mappPyr(mapper);
	Ptr<Map> mapPtr = mappPyr.calculate(img1, img2);

	// Print result
	MapAffine* mapAff = dynamic_cast<MapAffine*>(mapPtr.get());
	cout << endl << "--- Testing affine mapper ---" << endl;
	cout << Mat(linTr) << endl;
	cout << Mat(shift) << endl;
	cout << Mat(mapAff->getLinTr()) << endl;
	cout << Mat(mapAff->getShift()) << endl;

	// Display registration accuracy
	Mat dest;
	mapAff->inverseWarp(img2, dest);
	showDifference(img1, dest, DIFF_REGPIX_IM);

	waitKey(0);
	cvDestroyWindow(DIFF_IM);
	cvDestroyWindow(DIFF_REGPIX_IM);
}


/*
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SURF_FlannMatcher <img1> <img2>" << std::endl; }
