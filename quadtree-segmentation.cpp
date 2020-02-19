// quadtree-segmentation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <opencv2/imgproc.hpp>


#include <iostream>
#include <memory>
#include <cassert>


struct QuadTreeData
{
    size_t number{};
    float sum{};
    float sq_sum{};

    float getDeviation() const 
    { 
        return number ? sqrt((sq_sum - sum * sum / number) / number) : 0.f;
    }
};

inline QuadTreeData operator +(const QuadTreeData& left, const QuadTreeData& right)
{
    return { left.number + right.number, left.sum + right.sum, left.sq_sum + right.sq_sum };
}


struct QuadTree
{
    std::shared_ptr<QuadTreeData> data;
    std::unique_ptr<QuadTree> children[4]{};
};

bool canMerge(const std::unique_ptr<QuadTree>& left, const std::unique_ptr<QuadTree>& right)
{
    return !(left->children[0]) && !(right->children[0]) 
        && (*left->data + *right->data).getDeviation() <= 5.8;
}

void merge(std::unique_ptr<QuadTree>& left, std::unique_ptr<QuadTree>& right)
{
    *left->data = *left->data + *right->data;
    right->data = left->data;
}


std::unique_ptr<QuadTree> SplitQuadTree(const cv::Mat& src)
{
    QuadTreeData data{};

    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++)
        {
            const auto p = src.at<uchar>(i, j);
            ++data.number;
            data.sum += p;
            data.sq_sum += float(p) * p;
        }

    if (src.rows * src.cols <= 25 || data.getDeviation() <= 5.8)
        return std::make_unique<QuadTree>(QuadTree{ std::make_shared<QuadTreeData>(data) });

    const int w_med = src.cols / 2;
    const int h_med = src.rows / 2;

    return std::make_unique<QuadTree>(QuadTree{
        std::make_shared<QuadTreeData>(data),
        { 
            SplitQuadTree(src(cv::Rect(0, 0, w_med, h_med))), 
            SplitQuadTree(src(cv::Rect(w_med, 0, src.cols - w_med, h_med))),
            SplitQuadTree(src(cv::Rect(0, h_med, w_med, src.rows - h_med))),
            SplitQuadTree(src(cv::Rect(w_med, h_med, src.cols - w_med, src.rows - h_med))),
        }
     });

}


void MergeQuadTree(std::unique_ptr<QuadTree>& tree)
{
    if (!tree->children[0])
        return;
    const bool canRow1 = canMerge(tree->children[0], tree->children[1]);
    const bool canRow2 = canMerge(tree->children[2], tree->children[3]);
    if (canRow1 && canRow2)
    {
        merge(tree->children[0], tree->children[1]);
        merge(tree->children[2], tree->children[3]);
        return;
    }

    const bool canCol1 = canMerge(tree->children[0], tree->children[2]);
    const bool canCol2 = canMerge(tree->children[1], tree->children[3]);
    if (canCol1)
        merge(tree->children[0], tree->children[2]);
    if (canCol2)
        merge(tree->children[1], tree->children[3]);
    if (canCol1 && canCol2)
        return;

    if (!canCol1 && !canCol2)
    {
        if (canRow1)
            merge(tree->children[0], tree->children[1]);
        if (canRow2)
            merge(tree->children[2], tree->children[3]);
    }

    for (auto& child : tree->children)
    {
        MergeQuadTree(child);
    }
}


void OutputQuadTree(cv::Mat dst, const std::unique_ptr<QuadTree>& tree)
{
    if (!tree->children[0])
    {
        assert(tree->data->number);
        const auto val = tree->data->sum / tree->data->number;
        dst.setTo(cv::Scalar(val, val, val));
        return;
    }

    const int w_med = dst.cols / 2;
    const int h_med = dst.rows / 2;

    OutputQuadTree(dst(cv::Rect(0, 0, w_med, h_med)), tree->children[0]);
    OutputQuadTree(dst(cv::Rect(w_med, 0, dst.cols - w_med, h_med)), tree->children[1]);
    OutputQuadTree(dst(cv::Rect(0, h_med, w_med, dst.rows - h_med)), tree->children[2]);
    OutputQuadTree(dst(cv::Rect(w_med, h_med, dst.cols - w_med, dst.rows - h_med)), tree->children[3]);
}


int main(int argc, char** argv)
{
    using namespace cv;

    auto img = imread((argc > 1) ? argv[1] : samples::findFile("lena.jpg"), 0);

    // round (down) to the nearest power of 2 .. quadtree dimension is a pow of 2.
    int exponent = log(min(img.cols, img.rows)) / log(2);
    const int s = pow(2.0, (double)exponent);
    cv::resize(img, img, { s, s });

    namedWindow("original", WINDOW_AUTOSIZE);
    imshow("original", img);

    std::cout << "trying to split..\n";
    auto r = SplitQuadTree(img);

    std::cout << "splitted\n";
    Mat imgRect = img.clone();
    OutputQuadTree(imgRect, r);
    namedWindow("split", WINDOW_AUTOSIZE);
    imshow("split", imgRect);
    imwrite("split.jpg", imgRect);

    MergeQuadTree(r);
    Mat imgMerge = img.clone();
    OutputQuadTree(imgMerge, r);
    namedWindow("merge", WINDOW_AUTOSIZE);
    imshow("merge", imgMerge);
    imwrite("merge.jpg", imgMerge);

    while (true)
    {
        char c = (char)waitKey(10);
        if (c == 27) { break; }
    }
}
