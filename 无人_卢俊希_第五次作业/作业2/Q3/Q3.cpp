#include <osqp/osqp.h>
#include <iostream>
#include <cmath>
using namespace std;

int main()
{
    OSQPInt n = 2;
    OSQPInt m = 1;
    //x是非0元素的值
    // i是每个非0元素所在的行索引（从0开始）
    // p是每个x在i数组中的位置，长度等于列数+1，最后一个数字表示非零元素个数
    // P = diag(1, 10), CSC格式上三角
    OSQPFloat P_x[] = { 1.0, 10.0 };
    OSQPInt   P_i[] = { 0, 1 };
    OSQPInt   P_p[] = { 0, 1, 2 };

    // 一次项系数q = [-3, -30]
    OSQPFloat q[] = { -3.0, -30.0 };

    // A = [1, 1], CSC格式
    OSQPFloat A_x[] = { 1.0, 1.0 };
    OSQPInt   A_i[] = { 0, 0 };
    OSQPInt   A_p[] = { 0, 1, 2 };

    // 约束: -inf <= x+y <= 4
    OSQPFloat l[] = { -OSQP_INFTY };
    OSQPFloat u[] = { 4.0 };


    OSQPCscMatrix P_mat, A_mat;

    // P矩阵：2x2，2个非零元
    P_mat.m = n;    // 行 = 2
    P_mat.n = n;    // 列 = 2
    P_mat.nzmax = 2;//非零元素个数
    P_mat.nz = -1;   // -1 = CSC格式，必须有！
    P_mat.x = P_x;//指向数值数组
    P_mat.i = P_i;// 指向行索引数组
    P_mat.p = P_p;// 指向列指针数组

    // A矩阵：1x2，2个非零元
    A_mat.m = m;    // 行 = 1
    A_mat.n = n;    // 列 = 2
    A_mat.nzmax = 2;
    A_mat.nz = -1;   // -1 = CSC格式，必须有！
    A_mat.x = A_x;
    A_mat.i = A_i;
    A_mat.p = A_p;

    // 直接传参给 osqp_setup
    OSQPSettings settings;
    osqp_set_default_settings(&settings);
    settings.verbose = 1;//打开求解过程的日志输出

    OSQPSolver* solver = nullptr;

    //初始化求解器
    OSQPInt exitflag = osqp_setup(&solver,
        &P_mat, q,
        &A_mat, l, u,
        m, n,
        &settings);

    if (exitflag)
    {
        cerr << "[错误] osqp_setup 失败，错误码: "
            << exitflag << endl;
        return 1;
    }

    // 求解
    osqp_solve(solver);

    // 输出结果
    //OSQP_SOLVED是状态枚举值，表示成功找到最优解
    if (solver->info->status_val == OSQP_SOLVED)
    {
        double x_sol = solver->solution->x[0];//原问题最优解
        double y_sol = solver->solution->x[1];
        double mu = solver->solution->y[0]; // 拉格朗日乘子

        cout << "\n========== 求解结果 ==========" << endl;
        cout << "x        = " << x_sol << endl;
        cout << "y        = " << y_sol << endl;
        cout << "μ        = " << mu << endl;
        cout << "x + y    = " << x_sol + y_sol
            << "  (应 <= 4)" << endl;

        double obj = 0.5 * (x_sol - 3) * (x_sol - 3)
            + 5.0 * (y_sol - 3) * (y_sol - 3);
        cout << "f(x,y)   = " << obj << endl;
        cout << "================================\n";
    }
    else
    {
        cerr << "[警告] 求解状态: "
            << solver->info->status << endl;
    }

    // 清理（v1.0.0 只需要释放 solver）
    osqp_cleanup(solver);

    return 0;
}