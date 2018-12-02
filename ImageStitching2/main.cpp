#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <utility>
#include <cmath>
#include <ctime>
#include <io.h>

#define TOTAL_IMAGES 2
#define SHRINK_RATIO 0.5
#define FOCAL_LENGTH  1152.18    //576.09//1152.18//2304.36  //1152.18    //1145.87(set5)  //572.935  //704.5, 784.34(parrington)
#define ALPHA 0.04
#define THRESHOLD 450000000000.0    //450000000000.0   //800000000000.0 //35000000000.0
#define FILE_FOLDER "images2"
#define OUT_FILE "images7.jpg"
#define SHOWFEATURE false
#define SHOWLINE true
#define SHOWCYLIN false
#define SHOWSTITCH false

//set5:
//#define TOTAL_IMAGES 17
//#define SHRINK_RATIO 1  //no resize
//#define FOCAL_LENGTH  1145.87    
//#define ALPHA 0.04
//#define THRESHOLD 800000000000.0    
//#define FILE_FOLDER "set5"
//#define OUT_FILE "set5.jpg"

//parrington:
//#define TOTAL_IMAGES 18
//#define SHRINK_RATIO 1  //no resize
//#define FOCAL_LENGTH  784.34    
//#define ALPHA 0.04
//#define THRESHOLD 80000000000.0    
//#define FILE_FOLDER "parrington"
//#define OUT_FILE "parrington.jpg"

using namespace cv;
using namespace std;

typedef struct feature_point{
    int i;
    int j;
    vector<Vec3b> descriptor;
} Feature_Point;

vector<Mat> src(TOTAL_IMAGES);
vector<Mat> I(TOTAL_IMAGES);
vector<Mat> Ix(TOTAL_IMAGES);
vector<Mat> Iy(TOTAL_IMAGES);
vector<Mat> IxSq(TOTAL_IMAGES);
vector<Mat> IySq(TOTAL_IMAGES);
vector<Mat> Ixy(TOTAL_IMAGES);
vector<Mat> SxSq(TOTAL_IMAGES);
vector<Mat> SySq(TOTAL_IMAGES);
vector<Mat> Sxy(TOTAL_IMAGES);
vector<Mat> R;
vector<vector<Feature_Point>> features(TOTAL_IMAGES);
vector<vector<pair<Feature_Point,Feature_Point>>> good_matches(TOTAL_IMAGES);
vector<Mat> cyl;
vector<Mat> transformation;

void getAllFiles(string path, vector<string> &files);
int readImages();
void shrinkImages(const double m);
void intensity();
void gradient();
void products();
void gaussian();
void computeR(const double k);
void collect_fp();
void match_fp();
int calc_dist(const Feature_Point *a, const Feature_Point *b);
void cylindrical(bool do_cylindrical);
void ransac();
void align();
void normalization(vector<pair<int,int>> *uv, vector<pair<float,float>> *norm_uv, Mat *T);

int main(int argc, char *argv[])
{
    if(readImages() != 0){
        printf("Error reading image files!\n");
        return -1;
    };
    shrinkImages(SHRINK_RATIO);

    intensity();
    gradient();
    products();
    gaussian();

    computeR(ALPHA);

    collect_fp();

    match_fp();

    cylindrical(true);

    srand (time(NULL));
    ransac();

    align();

    return 0;
}

int readImages(){
    printf("Reading image files ...\n");

	string file_folder(FILE_FOLDER);
	vector<string> image_files;

	getAllFiles(file_folder, image_files);
	for (int i = 0; i < TOTAL_IMAGES; i++) {
		src[i] = imread(image_files[i].c_str(), CV_LOAD_IMAGE_COLOR); if (src[i].empty()) return -1;
	}
    return 0;
}

void getAllFiles(string path, vector<string> &files) {
	intptr_t hFile = 0;
	//文件信息    
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1) {
		do {
			if ((fileinfo.attrib &  _A_SUBDIR)) {
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					files.push_back(p.assign(path).append("\\").append(fileinfo.name));
					getAllFiles(p.assign(path).append("\\").append(fileinfo.name), files);
				}
			}
			else {
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}

		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}

}

void shrinkImages(const double m){
    printf("Resizing images to %fx ...\n", m);
    for(int i=0; i < TOTAL_IMAGES; i++){
        resize(src[i], src[i], Size((src[i]).cols * m, (src[i]).rows * m),0,0,INTER_LINEAR);
    }
}

void intensity(){
    printf("Convert to intensity ...\n");
    for(int i=0; i < TOTAL_IMAGES; i++){
        cvtColor( src[i], I[i], CV_BGR2GRAY );
    }
}

void gradient(){
    printf("Calculate x and y gradient ...\n");
    for(int i=0; i < TOTAL_IMAGES; i++){
        Sobel(I[i], Ix[i], CV_32F , 1, 0, 3, BORDER_DEFAULT);
        Sobel(I[i], Iy[i], CV_32F , 0, 1, 3, BORDER_DEFAULT);
    }
}

void products(){
    printf("Compute products of derivatives ...\n");
    for(int i=0; i < TOTAL_IMAGES; i++){
        pow(Ix[i], 2.0, IxSq[i]);
        pow(Iy[i], 2.0, IySq[i]);
        multiply(Ix[i], Iy[i], Ixy[i]);
    }
}

void gaussian(){
    printf("Compute gaussian sums ...\n");
    for(int i=0; i < TOTAL_IMAGES; i++){
        GaussianBlur(IxSq[i], SxSq[i], Size(7,7), 2.0, 0.0, BORDER_DEFAULT);
        GaussianBlur(IySq[i], SySq[i], Size(7,7), 0.0, 2.0, BORDER_DEFAULT);
        GaussianBlur(Ixy[i], Sxy[i], Size(7,7), 2.0, 2.0, BORDER_DEFAULT);
    }
}

void computeR(const double k){
    printf("Compute det, trace, R using k = %f ...\n", k);

    for(int i=0; i < TOTAL_IMAGES; i++){
        Mat tmp1, tmp2;
        multiply(SxSq[i], SySq[i], tmp1);
        multiply(Sxy[i], Sxy[i], tmp2);

        Mat traceM = (Mat)SxSq[i] + (Mat)SySq[i];
        pow(traceM, 2.0, traceM);

        Mat r = (tmp1 - tmp2) - k * traceM;
        R.push_back(r);
        //normalize( R[i], R[i], 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    }
    vector<Mat>().swap(Ix);
    vector<Mat>().swap(Iy);
    vector<Mat>().swap(IxSq);
    vector<Mat>().swap(IySq);
    vector<Mat>().swap(Ixy);
    vector<Mat>().swap(SxSq);
    vector<Mat>().swap(SySq);
    vector<Mat>().swap(Sxy);
}

void collect_fp(){
    printf("collect feature points ...\n");
    for(int z=0; z < TOTAL_IMAGES; z++){
        vector<Feature_Point> image_fp;
        for(int i = 2; i < I[z].rows-2 ; i++ ) {
            for(int j = 2; j < I[z].cols-2 ; j++ ) {
                if( R[z].at<float>(i,j) > THRESHOLD ) {
                    if(R[z].at<float>(i,j) > R[z].at<float>(i-1,j-1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i,j-1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i+1,j-1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i-1,j) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i+1,j) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i-1,j+1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i,j+1) &&
                       R[z].at<float>(i,j) > R[z].at<float>(i+1,j+1) ){

                        Feature_Point fp;
                        fp.i = i;
                        fp.j = j;
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-2,j-2));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-2,j-1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-2,j));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-2,j+1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-2,j+2));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-1,j-2));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-1,j-1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-1,j));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-1,j+1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i-1,j+2));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i,j-2));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i,j-1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i,j));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i,j+1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i,j+2));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+1,j-2));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+1,j-1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+1,j));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+1,j+1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+1,j+2));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+2,j-2));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+2,j-1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+2,j));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+2,j+1));
                        fp.descriptor.push_back(src[z].at<Vec3b>(i+2,j+2));

                        image_fp.push_back(fp);
                    }
                }
            }
        }

        features[z] = image_fp;
    }
}

void match_fp(){
    printf("matching feature points ...\n");

    for(int z=0; z < TOTAL_IMAGES-1; z++){
        printf("features in %d: %d\n", z, features[z].size());
        printf("features in %d: %d\n", z+1, features[z+1].size());

        Mat window1 = src[z].clone();
        Mat window2 = src[z+1].clone();
        for(int i=0; i < features[z].size(); i++){
            circle( window1, Point( features[z][i].j, features[z][i].i ), 5, Scalar(0,255,255), 2, 8, 0 );
        }
        for(int i=0; i < features[z+1].size(); i++){
            circle( window2, Point( features[z+1][i].j, features[z+1][i].i ), 5, Scalar(0, 255, 255), 2, 8, 0 );
        }

        vector<pair<int,int>> matchingpoints;
        int min_dist = 2147483647;
        for(int i=0; i < features[z].size(); i++){
            int dist = 2147483647;
            int point_b = -1;
            for(int j=0; j < features[z+1].size(); j++){
                /** cheating upon the knowledge of left-right image sequence **/
                if(features[z+1][j].j > src[z+1].cols*2/3 || features[z+1][j].j > features[z][i].j){
                    continue;
                }
                int d = calc_dist(&features[z][i], &features[z+1][j]);
                if(d < dist){
                    dist = d;
                    point_b = j;
                }
            }
            if( dist < min_dist ) min_dist = dist;
            matchingpoints.push_back(make_pair(point_b, dist));

        }
        printf("min_dist: %d\n", min_dist);

        vector<int> point_b_min_dist(features[z+1].size() , 2147483647);
        for(int i=0; i < features[z].size(); i++){
            if( matchingpoints[i].second <= max(2*min_dist, 60000)){
                if(matchingpoints[i].second < point_b_min_dist[matchingpoints[i].first]){
                    point_b_min_dist[matchingpoints[i].first] = matchingpoints[i].second;
                }
            }
        }

        for(int i=0; i < features[z].size(); i++){
            if( matchingpoints[i].second <= max(2*min_dist, 60000)){
                if(matchingpoints[i].second > point_b_min_dist[matchingpoints[i].first]){
                    continue;
                }
                if(features[z][i].j < src[z].cols*1/3){ continue; }
                good_matches[z].push_back(make_pair(features[z][i], features[z+1][matchingpoints[i].first]));
                circle( window1, Point( features[z][i].j, features[z][i].i ), 5, Scalar(20,20,255), 2, 8, 0 );
                circle( window2, Point( features[z+1][matchingpoints[i].first].j, features[z+1][matchingpoints[i].first].i ), 5, Scalar(20,20,255), 2, 8, 0 );
            }
        }

        /** show matched feature**/
		if (SHOWFEATURE) {
			char windowname[10]; char windowname2[10];
			sprintf(windowname, "img%d", z);
			namedWindow(windowname, CV_WINDOW_NORMAL);
			imshow(windowname, window1);
			sprintf(windowname2, "img%d", z + 1);
			namedWindow(windowname2, CV_WINDOW_NORMAL);
			imshow(windowname2, window2);
			waitKey(0);
			destroyWindow(windowname);
			destroyWindow(windowname2);
		}
		

        /** show feature matching line**/
		if (SHOWLINE) {
			Mat matching_img(window1.rows * 2, window1.cols, window1.type());
			Mat part;
			part = matching_img(Rect(0, 0, window1.cols, window1.rows));
			window1.copyTo(part);
			part = matching_img(Rect(0, window1.rows, window1.cols, window1.rows));
			window2.copyTo(part);

			for (int i = 0; i < good_matches[z].size(); i++) {
				line(matching_img, Point(good_matches[z][i].first.j, good_matches[z][i].first.i),
					Point(good_matches[z][i].second.j, good_matches[z][i].second.i + window1.rows), Scalar(255,255,0), 2);
			}

			namedWindow("matching img", CV_WINDOW_NORMAL);
			imshow("matching img", matching_img);
			waitKey(0);
			destroyWindow("matching img");
		}
    }
}

int calc_dist(const Feature_Point *a, const Feature_Point *b){
    int d = 0;
    for(int i=0; i < 25; i++){
        int tmp = (a -> descriptor[i])[0] - (b -> descriptor[i])[0];
        d += tmp * tmp;
        tmp = (a -> descriptor[i])[1] - (b -> descriptor[i])[1];
        d += tmp * tmp;
        tmp = (a -> descriptor[i])[2] - (b -> descriptor[i])[2];
        d += tmp * tmp;
    }
    return d;
}

void cylindrical(bool do_cylindrical){
    printf("cylindrical projection ...\n");
    if(!do_cylindrical){
        for(int z=0; z < TOTAL_IMAGES; z++){
            cyl.push_back(src[z]);
        }
        return;
    }

    for(int z=0; z < TOTAL_IMAGES; z++){
        int rows = src[z].rows;
        int cols = src[z].cols;
        Mat m_cyl(rows, cols, CV_8UC3, Scalar(0, 0, 0));

        for(int i=0; i < rows; i++){
            for(int j=0; j < cols; j++){
                int y = i - (rows/2);
                int x = j - (cols/2);
                int x_cyl = FOCAL_LENGTH * atan( x / FOCAL_LENGTH );
                int y_cyl = FOCAL_LENGTH * y / sqrt( x*x + FOCAL_LENGTH*FOCAL_LENGTH );
				//int y_cyl = sqrt(x*x + FOCAL_LENGTH * FOCAL_LENGTH) * y / FOCAL_LENGTH;
				//int y_cyl = FOCAL_LENGTH * FOCAL_LENGTH / sqrt(x*x + FOCAL_LENGTH * FOCAL_LENGTH) * atan(y / FOCAL_LENGTH);
                m_cyl.at<Vec3b>(y_cyl + rows/2, x_cyl + cols/2) = src[z].at<Vec3b>(i,j);
            }
        }
        cyl.push_back(m_cyl);

        if(z < TOTAL_IMAGES-1){
            /** project the matching feature points to cylinder**/
            for(int i=0; i < good_matches[z].size(); i++){
                int y = good_matches[z][i].first.i - (rows/2);
                int x = good_matches[z][i].first.j - (cols/2);
                good_matches[z][i].first.j = FOCAL_LENGTH * atan( x / FOCAL_LENGTH ) + cols/2;
                good_matches[z][i].first.i = FOCAL_LENGTH * y / sqrt( x*x + FOCAL_LENGTH*FOCAL_LENGTH ) + rows/2;
				//good_matches[z][i].first.i = sqrt(x*x + FOCAL_LENGTH * FOCAL_LENGTH) * y / FOCAL_LENGTH + rows / 2;
				//good_matches[z][i].first.i = FOCAL_LENGTH * FOCAL_LENGTH / sqrt(x*x + FOCAL_LENGTH * FOCAL_LENGTH) * atan(y / FOCAL_LENGTH) + rows / 2;

                y = good_matches[z][i].second.i - (rows/2);
                x = good_matches[z][i].second.j - (cols/2);
                good_matches[z][i].second.j = FOCAL_LENGTH * atan( x / FOCAL_LENGTH ) + cols/2;
                good_matches[z][i].second.i = FOCAL_LENGTH * y / sqrt( x*x + FOCAL_LENGTH*FOCAL_LENGTH ) + rows/2;
				//good_matches[z][i].second.i = sqrt(x*x + FOCAL_LENGTH * FOCAL_LENGTH) * y / FOCAL_LENGTH + rows / 2;
				//good_matches[z][i].second.i = FOCAL_LENGTH * FOCAL_LENGTH / sqrt(x*x + FOCAL_LENGTH * FOCAL_LENGTH) * atan(y / FOCAL_LENGTH) + rows / 2;
            }
        }

        /** show cylindrical projected image**/
		if (SHOWCYLIN) {
			char windowname[10];
			sprintf(windowname, "cyl%d", z);
			namedWindow(windowname, CV_WINDOW_FULLSCREEN);
			imshow(windowname, m_cyl);
			waitKey(0);
		}
		
    }
}

void ransac(){
    printf("ransac ...\n");
    for(int z=0; z < TOTAL_IMAGES-1; z++){
        int max_inliers = -1;
        Mat bestModel;
        for(int k=0; k < 72; k++){
            int match_size = good_matches[z].size();
            int a,b,c,d;
            a = rand() % match_size;
            do {b = rand() % match_size;} while (a == b);
            do {c = rand() % match_size;} while (a == c || b == c);
            do {d = rand() % match_size;} while (a == d || b == d || c == d);

            /** finding homography of four points by implementing normalized DLT
            *** but sfm::normalizePoints not available on Windows
            **/
            /*vector<pair<int,int>> uv;
            uv.push_back(make_pair(good_matches[z][a].second.i,good_matches[z][a].second.j));
            uv.push_back(make_pair(good_matches[z][b].second.i,good_matches[z][b].second.j));
            uv.push_back(make_pair(good_matches[z][c].second.i,good_matches[z][c].second.j));
            uv.push_back(make_pair(good_matches[z][d].second.i,good_matches[z][d].second.j));
            vector<pair<float,float>> norm_uv;
            Mat T(3, 3, CV_32F, Scalar(0));
            normalization(&uv, &norm_uv, &T);

            vector<pair<int,int>> u_v_;
            u_v_.push_back(make_pair(good_matches[z][a].first.i,good_matches[z][a].first.j));
            u_v_.push_back(make_pair(good_matches[z][b].first.i,good_matches[z][b].first.j));
            u_v_.push_back(make_pair(good_matches[z][c].first.i,good_matches[z][c].first.j));
            u_v_.push_back(make_pair(good_matches[z][d].first.i,good_matches[z][d].first.j));
            vector<pair<float,float>> norm_u_v_;
            Mat Tp(3, 3, CV_32F, Scalar(0));
            normalization(&u_v_, &norm_u_v_, &Tp);

            //sfm::normalizePoints(uv,norm_uv,T);
            //sfm::normalizePoints(u_v_,norm_u_v_,Tp);

            Mat A(8, 9, CV_32F, Scalar(0.0));
            int u[] = {norm_uv[0].first,norm_uv[1].first,norm_uv[2].first,norm_uv[3].first};
            int v[] = {norm_uv[0].second,norm_uv[1].second,norm_uv[2].second,norm_uv[3].second};
            int u_new[] = {norm_u_v_[0].first,norm_u_v_[1].first,norm_u_v_[2].first,norm_u_v_[3].first};
            int v_new[] = {norm_u_v_[0].second,norm_u_v_[1].second,norm_u_v_[2].second,norm_u_v_[3].second};

            for(int i=0; i < 4; i++){
                A.at<float>(i*2,3) = -u[i];    A.at<float>(i*2,4) = -v[i];    A.at<float>(i*2,5) = -1;
                A.at<float>(i*2,6) = v_new[i]*u[i];
                A.at<float>(i*2,7) = v_new[i]*v[i];
                A.at<float>(i*2,8) = v_new[i];
                A.at<float>(i*2+1,0) = u[i];    A.at<float>(i*2+1,1) = v[i];    A.at<float>(i*2+1,2) = 1;
                A.at<float>(i*2+1,6) = -u_new[i]*u[i];
                A.at<float>(i*2+1,7) = -u_new[i]*v[i];
                A.at<float>(i*2+1,8) = -u_new[i];
            }

            Mat h;

            SVD::solveZ(A,h);

            Mat H(3, 3, CV_32F, Scalar(0));
            H.at<float>(0,0) = h.at<float>(0,0);  H.at<float>(0,1) = h.at<float>(1,0);  H.at<float>(0,2) = h.at<float>(2,0);
            H.at<float>(1,0) = h.at<float>(3,0);  H.at<float>(1,1) = h.at<float>(4,0);  H.at<float>(1,2) = h.at<float>(5,0);
            H.at<float>(2,0) = h.at<float>(6,0);  H.at<float>(2,1) = h.at<float>(7,0);  H.at<float>(2,2) = h.at<float>(8,0);

            Mat model;
            model = Tp.inv() * H;
            model = model * T;*/

            /** give up finding homography of four points by implementing normalized DLT **/
            vector<Point> uv;
            uv.push_back(Point(good_matches[z][a].first.i,good_matches[z][a].first.j));
            uv.push_back(Point(good_matches[z][b].first.i,good_matches[z][b].first.j));
            uv.push_back(Point(good_matches[z][c].first.i,good_matches[z][c].first.j));
            uv.push_back(Point(good_matches[z][d].first.i,good_matches[z][d].first.j));
            vector<Point> u_v_;
            u_v_.push_back(Point(good_matches[z][a].second.i,good_matches[z][a].second.j));
            u_v_.push_back(Point(good_matches[z][b].second.i,good_matches[z][b].second.j));
            u_v_.push_back(Point(good_matches[z][c].second.i,good_matches[z][c].second.j));
            u_v_.push_back(Point(good_matches[z][d].second.i,good_matches[z][d].second.j));
            Mat model = estimateRigidTransform(u_v_,uv,true);
            if(model.empty()){
                continue;
            }

            model.convertTo(model, CV_32F);

            /** count inliers **/
            int inliers = 0;
            for(int i=0; i < match_size; i++){
                Mat m1(3, 1, CV_32F, Scalar(0));
                m1.at<float>(0,0) = (float) good_matches[z][i].second.i;
                m1.at<float>(1,0) = (float) good_matches[z][i].second.j;
                m1.at<float>(2,0) = 1.0;
                Mat m2(3, 1, CV_32F, Scalar(0));
                m2 = model * m1;

                float d1 = m2.at<float>(0,0) - good_matches[z][i].first.i;
                float d2 = m2.at<float>(1,0) - good_matches[z][i].first.j;
                if(d1*d1+d2*d2 < 18){
                    inliers++;
                }
            }

            if(inliers > max_inliers){
                max_inliers = inliers;
                bestModel = model.clone();
            }

        }
        printf("max_inliers[%d]:%d\n",z,max_inliers);
        printf("  %f %f %f\n  %f %f %f\n",
               bestModel.at<float>(0,0),bestModel.at<float>(0,1),bestModel.at<float>(0,2),
               bestModel.at<float>(1,0),bestModel.at<float>(1,1),bestModel.at<float>(1,2));
        transformation.push_back(bestModel);
    }
}

void align(){
    printf("alignment ...\n");
    Vec3b black(0,0,0);

    Mat stitched(cyl[0].rows*2, cyl[0].cols * TOTAL_IMAGES * 3/5, CV_8UC3, Scalar(0,0,0));
    cyl[0].copyTo( stitched( Rect(0, cyl[0].rows/2, cyl[0].cols, cyl[0].rows) ) );

    for(int z = 1; z < TOTAL_IMAGES; z++){
        printf("  stitching image %d/%d ...\n",z, TOTAL_IMAGES-1);

        for(int i=0; i < cyl[z].rows; i++){
            for(int j=0; j < cyl[z].cols; j++){
                if(cyl[z].at<Vec3b>(i,j) == black ){ continue; }

                Mat m1(3, 1, CV_32F, Scalar(0));
                m1.at<float>(0,0) = (float) i;
                m1.at<float>(1,0) = (float) j;
                m1.at<float>(2,0) = 1.0;
                Mat m2(3, 1, CV_32F, Scalar(0));

                for(int k = z-1; k >= 0; k--){
                    m2 = transformation[k] * m1;
                    m1.at<float>(0,0) = m2.at<float>(0,0);
                    m1.at<float>(1,0) = m2.at<float>(1,0);
                    m1.at<float>(2,0) = 1.0;
                }

                int new_i = (int) (m2.at<float>(0,0)) + cyl[0].rows/2;
                int new_j = (int) (m2.at<float>(1,0));

                if(new_i >= 0 && new_i < stitched.rows-1 && new_j >= 0 && new_j < stitched.cols-1){
                    if(stitched.at<Vec3b>(new_i, new_j) != black){
                        /** overlapping, blend pixels **/
                        double alpha, beta;
                        alpha = (double)(j) / (double)(cyl[z].cols/2);
                        alpha = alpha > 1.0 ? 1.0 : alpha;
                        beta = 1.0 - alpha;
                        addWeighted( cyl[z].at<Vec3b>(i,j), alpha, stitched.at<Vec3b>(new_i, new_j), beta, 0.0, stitched.at<Vec3b>(new_i, new_j));
                    } else {
                        stitched.at<Vec3b>(new_i, new_j) = cyl[z].at<Vec3b>(i,j);
                        stitched.at<Vec3b>(new_i+1, new_j) = cyl[z].at<Vec3b>(i,j);
                        stitched.at<Vec3b>(new_i, new_j+1) = cyl[z].at<Vec3b>(i,j);
                    }
                }
            }
        }

        /** show stitched so far **/
		if (SHOWSTITCH) {
			namedWindow("panorama", CV_WINDOW_NORMAL);
			imshow("panorama", stitched);
			waitKey(0);
		}
		
    }
    int top_bound = 0;
    for(int i=0; i < stitched.rows; i++){
        for(int j=0; j < stitched.cols; j++){
            if(stitched.at<Vec3b>(i,j) != black){
                top_bound = i;
                break;
            }
        }
        if(top_bound != 0){
            break;
        }
    }
    int bottom_bound = stitched.rows-1;
    for(int i=stitched.rows-1; i >= 0; i--){
        for(int j=0; j < stitched.cols; j++){
            if(stitched.at<Vec3b>(i,j) != black){
                bottom_bound = i;
                break;
            }
        }
        if(bottom_bound != stitched.rows-1){
            break;
        }
    }
    int right_bound = stitched.cols-1;
    for(int j=stitched.cols-1; j >= 0; j--){
        for(int i=0; i < stitched.rows; i++){
            if(stitched.at<Vec3b>(i,j) != black){
                right_bound = j;
                break;
            }
        }
        if(right_bound != stitched.cols-1){
            break;
        }
    }
    stitched = stitched(Rect(0,top_bound,right_bound,bottom_bound-top_bound));

    imwrite("panorama_cropped.jpg",stitched);  //

    /** warpAffine **/
    Point2f srcTri[3];
    Point2f dstTri[3];
    srcTri[0] = Point2f( 0,cyl[0].rows/2 - top_bound);

    Mat m1(3, 1, CV_32F, Scalar(0));
    m1.at<float>(0,0) = 0;
    m1.at<float>(1,0) = cyl[TOTAL_IMAGES-1].cols-1;
    m1.at<float>(2,0) = 1.0;
    Mat m2(3, 1, CV_32F, Scalar(0));
    for(int k = TOTAL_IMAGES-2; k >= 0; k--){
        m2 = transformation[k] * m1;
        m1.at<float>(0,0) = m2.at<float>(0,0);
        m1.at<float>(1,0) = m2.at<float>(1,0);
        m1.at<float>(2,0) = 1.0;
    }
    srcTri[1] = Point2f( (m2.at<float>(1,0)), (m2.at<float>(0,0)) + cyl[0].rows/2 - top_bound);

    m1.at<float>(0,0) = cyl[TOTAL_IMAGES-1].rows-1;
    m1.at<float>(1,0) = cyl[TOTAL_IMAGES-1].cols-1;
    m1.at<float>(2,0) = 1.0;
    for(int k = TOTAL_IMAGES-2; k >= 0; k--){
        m2 = transformation[k] * m1;
        m1.at<float>(0,0) = m2.at<float>(0,0);
        m1.at<float>(1,0) = m2.at<float>(1,0);
        m1.at<float>(2,0) = 1.0;
    }
    srcTri[2] = Point2f( (m2.at<float>(1,0)), (m2.at<float>(0,0)) + cyl[0].rows/2  - top_bound);

    dstTri[0] = Point2f( 0 ,cyl[0].rows/2 - top_bound);
    dstTri[1] = Point2f( stitched.cols-1 ,cyl[0].rows/2 - top_bound );
    dstTri[2] = Point2f( stitched.cols-1 ,stitched.rows - 1);

    Mat warp_mat( 2, 3, CV_32FC1 );
    warp_mat = getAffineTransform( srcTri, dstTri );
    warpAffine( stitched, stitched, warp_mat, stitched.size() );

    imwrite("panorama_wrap.jpg",stitched); //

    stitched = stitched(Rect(0, cyl[0].rows/2 - top_bound, stitched.cols, stitched.rows - (cyl[0].rows/2 - top_bound) ));

    imwrite(OUT_FILE,stitched);
}

void normalization(vector<pair<int,int>> *uv, vector<pair<float,float>> *norm_uv, Mat *T){
    float meanx = 0, meany = 0;
    for(int i=0; i < uv->size(); i++){
        meanx += uv->at(i).first;
        meany += uv->at(i).second;
    }
    meanx /= uv->size();
    meany /= uv->size();
    float value = 0;
    for(int i=0; i<4; i++){
        value += sqrt(pow(uv->at(i).first - meanx, 2.0) + pow(uv->at(i).second - meany, 2.0));
    }
    value /= uv->size();

    float scale = sqrt(2.0)/value;
    float tx = -scale * meanx;
    float ty = -scale * meany;

    T->at<float>(0,0) = scale;                              T->at<float>(0,2) = tx;
                                T->at<float>(1,1) = scale;  T->at<float>(1,2) = ty;
                                                            T->at<float>(2,2) = 1.0;

    for(int i=0; i < uv->size(); i++){
        Mat x(3, 1, CV_32F, Scalar(0));
        x.at<float>(0,0) = uv->at(i).first;
        x.at<float>(1,0) = uv->at(i).second;
        x.at<float>(2,0) = 1.0;
        Mat xp(3, 1, CV_32F, Scalar(0));
        xp = *T * x;
        norm_uv->push_back(make_pair(xp.at<float>(0,0)/xp.at<float>(2,0), xp.at<float>(1,0)/xp.at<float>(2,0)));
    }
}
