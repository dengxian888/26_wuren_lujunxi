#include<Eigen/Dense>
#include <iostream>
using namespace std;

int main()
{
    Eigen::Vector2d X(0.0,0.0);//创建一个二维向量，作为初始坐标
    double eta = 0.1;//学习率
    int max_iter = 10000;//最大迭代次数
    double tol = 1e-3;//收敛条件：距离目标点的距离小于0.001就停止

    for (int k = 0; k < max_iter; k++)
    {
        Eigen::Vector2d grad;
        grad(0) = X(0) - 3;
        grad(1) = 10 * (X(1) - 3);

        X = X - eta * grad;

        if ((X - Eigen::Vector2d(3.0, 3.0)).norm() < tol)
        {
            cout << "第" << k + 1 << "次迭代后收敛" << endl;
            break;
        }
    }
}
