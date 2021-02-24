#include "descriptor_matcher.h"

#include <opencv2/flann/miniflann.hpp>
#include "flann_factory.h"

void phg::DescriptorMatcher::filterMatchesRatioTest(const std::vector<std::vector<cv::DMatch>> &matches,
                                                    std::vector<cv::DMatch> &filtered_matches)
{
    filtered_matches.clear();

    for (std::vector<cv::DMatch> best_k_match : matches) {
        //расстояние до первой ближайшей сильно меньше расстояния до второй ближайшей
        if (best_k_match[0].distance < 0.6 * best_k_match[1].distance) {
            filtered_matches.push_back(best_k_match[0]);
        }
    }
}


void phg::DescriptorMatcher::filterMatchesClusters(const std::vector<cv::DMatch> &matches,
                                                   const std::vector<cv::KeyPoint> keypoints_query,
                                                   const std::vector<cv::KeyPoint> keypoints_train,
                                                   std::vector<cv::DMatch> &filtered_matches)
{
    filtered_matches.clear();

    const size_t  total_neighbours  = 5;  // total number of neighbours to test (including candidate)
    const size_t  consistent_matches  = 3;  // minimum number of consistent matches (including candidate)
    const float  radius_limit_scale  = 2.f;  // limit search radius by scaled median

    const int n_matches = matches.size();

    if (n_matches < total_neighbours) {
        throw std::runtime_error("DescriptorMatcher::filterMatchesClusters : too few matches");
    }

    cv::Mat points_query(n_matches, 2, CV_32FC1);
    cv::Mat points_train(n_matches, 2, CV_32FC1);
    for (int i = 0; i < n_matches; ++i) {
        points_query.at<cv::Point2f>(i) = keypoints_query[matches[i].queryIdx].pt;
        points_train.at<cv::Point2f>(i) = keypoints_train[matches[i].trainIdx].pt;
    }

    // размерность всего 2, так что точное KD-дерево
    std::shared_ptr<cv::flann::IndexParams> index_params = flannKdTreeIndexParams(1);
    std::shared_ptr<cv::flann::SearchParams> search_params = flannKsTreeSearchParams(n_matches - 1);

    std::shared_ptr<cv::flann::Index> index_query = flannKdTreeIndex(points_query, index_params);
    std::shared_ptr<cv::flann::Index> index_train = flannKdTreeIndex(points_train, index_params);

    // для каждой точки найти total neighbors ближайших соседей
    cv::Mat indices_query(n_matches, total_neighbours, CV_32SC1);
    cv::Mat distances_query(n_matches, total_neighbours, CV_32FC1);
    cv::Mat indices_train(n_matches, total_neighbours, CV_32SC1);
    cv::Mat distances_train(n_matches, total_neighbours, CV_32FC1);

    index_query->knnSearch(points_query, indices_query, distances_query, total_neighbours, *search_params);
    index_train->knnSearch(points_train, indices_train, distances_train, total_neighbours, *search_params);

    // оценить радиус поиска для каждой картинки
    // NB: radius_query, radius_train: радиусы!
    float radius_query, radius_train;
    {
        std::vector<double> max_dists_query(n_matches);
        std::vector<double> max_dists_train(n_matches);
        for (int i = 0; i < n_matches; ++i) {
            //Надо ли тут возводить расстояние в квадрат?
            max_dists_query[i] = distances_query.at<float>(i, total_neighbours - 1);
            max_dists_train[i] = distances_train.at<float>(i, total_neighbours - 1);
        }

        int median_pos = n_matches / 2;
        std::nth_element(max_dists_query.begin(), max_dists_query.begin() + median_pos, max_dists_query.end());
        std::nth_element(max_dists_train.begin(), max_dists_train.begin() + median_pos, max_dists_train.end());

        radius_query = max_dists_query[median_pos] * radius_limit_scale;
        radius_train = max_dists_train[median_pos] * radius_limit_scale;
    }

    // метч остается, если левое и правое множества первых total_neighbors соседей в радиусах поиска(radius2_query, radius2_train) имеют как минимум consistent_matches общих элементов
    // TODO заполнить filtered_matches
    for (int i = 0; i < n_matches; ++i) {
        std::vector<int> neighbors_indices_query_i;
        std::vector<int> neighbors_indices_train_i;

        int j = 0;
        while (neighbors_indices_query_i.size() < total_neighbours && distances_query.at<float>(i, j) < radius_query) {
            neighbors_indices_query_i.push_back(indices_query.at<int>(i, j));
            ++j;
        }
        j = 0;
        while (neighbors_indices_train_i.size() < total_neighbours && distances_train.at<float>(i, j) < radius_train) {
            neighbors_indices_train_i.push_back(indices_train.at<int>(i, j));
            ++j;
        }

        int num_matches = 0;
        for (auto& qi : neighbors_indices_query_i) {
            for (auto& ti : neighbors_indices_train_i) {
                if (qi == ti) {
                    ++num_matches;
                }
            }
        }

        if (num_matches >= consistent_matches) {
            filtered_matches.push_back(matches[i]);
        }
    }
}