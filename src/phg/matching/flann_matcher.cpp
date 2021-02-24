#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(10);
    search_params = flannKsTreeSearchParams(10);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    const int ndesc = query_desc.rows;

    matches.resize(ndesc);

    cv::Mat indices(ndesc, k, CV_32SC1);
    cv::Mat dists(ndesc, k, CV_32FC1);
    flann_index->knnSearch(query_desc, indices, dists, k, *search_params);


    for (int qi = 0; qi < ndesc; ++qi) {
        std::vector<cv::DMatch> &dst = matches[qi];
        dst.clear();
        for (int ni = 0; ni < k; ++ni) {
            cv::DMatch match;
            match.distance = dists.at<float>(qi, ni);
            match.imgIdx = 0;
            match.queryIdx = qi;
            match.trainIdx = indices.at<int>(qi, ni);
            dst.push_back(match);
        }
    }
}
