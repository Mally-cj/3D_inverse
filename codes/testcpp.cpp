#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <algorithm>

// 简单二维坐标
struct etGridCrd {
    int line;
    int cmp;
    etGridCrd(int l=0, int c=0) : line(l), cmp(c) {}
    bool operator<(const etGridCrd &b) const {
        return line == b.line ? cmp < b.cmp : line < b.line;
    }
    bool operator==(const etGridCrd &b) const {
        return line == b.line && cmp == b.cmp;
    }
};

// 简单二维点
struct GPoint {
    double x, y;
    GPoint(double x_=0, double y_=0) : x(x_), y(y_) {}
    double GetX() const { return x; }
    double GetY() const { return y; }
};

// 网格边界辅助
struct GridRange {
    int lineStart, lineEnd, cmpStart, cmpEnd;
    GridRange(int ls, int le, int cs, int ce) : lineStart(ls), lineEnd(le), cmpStart(cs), cmpEnd(ce) {}
    bool IsValidLineCmpNo(const etGridCrd &crd) const {
        return crd.line >= lineStart && crd.line <= lineEnd && crd.cmp >= cmpStart && crd.cmp <= cmpEnd;
    }
    int GetLineStart() const { return lineStart; }
    int GetLineEnd() const { return lineEnd; }
    int GetCmpStart() const { return cmpStart; }
    int GetCmpEnd() const { return cmpEnd; }
};

// 简单的插值，线性采样剖面
namespace etDataTraverse {
std::vector<etGridCrd> CalcGridPos(const GridRange &range, const std::vector<GPoint> &points) {
    std::vector<etGridCrd> result;
    for (size_t i = 1; i < points.size(); ++i) {
        GPoint p0 = points[i-1];
        GPoint p1 = points[i];
        int steps = std::max(std::abs(int(p1.x - p0.x)), std::abs(int(p1.y - p0.y)));
        steps = std::max(steps, 1);
        for (int s = 0; s <= steps; ++s) {
            double t = s / double(steps);
            int x = int(round(p0.x + t * (p1.x - p0.x)));
            int y = int(round(p0.y + t * (p1.y - p0.y)));
            etGridCrd crd(y, x);
            if (range.IsValidLineCmpNo(crd))
                result.push_back(crd);
        }
    }
    return result;
}
}

// 主类
struct etWIInversionPredictJob {
    std::vector<etGridCrd> _vWellPos;
    std::map<etGridCrd, float> _vWellMask;
    GridRange _maxGridRange;

    etWIInversionPredictJob(const std::vector<etGridCrd>& wpos, const GridRange& range)
        : _vWellPos(wpos), _maxGridRange(range)
    {
        GenerateWellMask();
    }

    void GenerateWellMask() {
        int wellRange = 15;
        for (size_t i = 0; i < _vWellPos.size(); i++) {
            for (int iline = -wellRange; iline <= wellRange; iline++)
                for (int icmp = -wellRange; icmp <= wellRange; icmp++) {
                    etGridCrd crd(_vWellPos[i].line + iline, _vWellPos[i].cmp + icmp);
                    float weight = std::exp(-(iline * iline + icmp * icmp) / 50.0f);
                    if (!_vWellMask.count(crd))
                        _vWellMask[crd] = weight;
                    else if (_vWellMask[crd] < weight)
                        _vWellMask[crd] = weight;
                }
        }
    }

    // 生成三井剖面
    bool GetWellLine(std::vector<etGridCrd>& vCrd, std::vector<float>& vMask) {
        if (_vWellPos.size() < 3) return false;
        std::vector<int> idx;
        while (idx.size() < 3) {
            int tmp = rand() % _vWellPos.size();
            if (std::find(idx.begin(), idx.end(), tmp) == idx.end())
                idx.push_back(tmp);
        }

        GPoint p01, p12;
        double L0, L1, L2;
        do {
            auto dist2 = [&](int a, int b) {
                return (_vWellPos[a].cmp - _vWellPos[b].cmp) * (_vWellPos[a].cmp - _vWellPos[b].cmp)
                     + (_vWellPos[a].line - _vWellPos[b].line) * (_vWellPos[a].line - _vWellPos[b].line);
            };
            double d01 = dist2(idx[0], idx[1]);
            double d12 = dist2(idx[1], idx[2]);
            double d20 = dist2(idx[2], idx[0]);
            if (d01 <= d20 && d20 <= d12) idx = {idx[1], idx[0], idx[2]};
            else if (d12 <= d01 && d01 <= d20) idx = {idx[2], idx[1], idx[0]};
            else if (d12 <= d20 && d20 <= d01) idx = {idx[1], idx[2], idx[0]};
            else if (d20 <= d01 && d01 <= d12) idx = {idx[2], idx[0], idx[1]};
            else if (d20 <= d12 && d12 <= d01) idx = {idx[0], idx[2], idx[1]};

            double pi = M_PI, pi2 = M_PI_2;
            auto atan2d = [](int y, int x) { return atan2(y, x); };
            double n10 = atan2d(_vWellPos[idx[0]].line - _vWellPos[idx[1]].line, _vWellPos[idx[0]].cmp - _vWellPos[idx[1]].cmp) + pi;
            double n12 = atan2d(_vWellPos[idx[2]].line - _vWellPos[idx[1]].line, _vWellPos[idx[2]].cmp - _vWellPos[idx[1]].cmp) + pi;
            // 归一到-pi~pi
            if (n10 > pi) n10 -= pi * 2;
            if (n12 > pi) n12 -= pi * 2;
            if (n10 - n12 > pi) n10 -= pi * 2;
            else if (n10 - n12 < -pi) n12 -= pi * 2;
            // 1边向量
            L1 = (n12 - n10) * (1.0 * rand() / RAND_MAX) + n10;
            if (L1 < -pi) L1 += pi * 2;
            else if (L1 > pi) L1 -= pi * 2;
            // 01
            double L01 = atan2d(_vWellPos[idx[1]].line - _vWellPos[idx[0]].line, _vWellPos[idx[1]].cmp - _vWellPos[idx[0]].cmp);
            double n1 = L1 + pi2;
            while (L01 - n1 > pi2) L01 -= pi;
            while (L01 - n1 < -pi2) n1 -= pi;
            L0 = (n1 - L01) * (1.0 * rand() / RAND_MAX) + L01;
            if (L0 < -pi) L0 += pi * 2;
            else if (L0 > pi) L0 -= pi * 2;
            // 21
            double L21 = atan2d(_vWellPos[idx[1]].line - _vWellPos[idx[2]].line, _vWellPos[idx[1]].cmp - _vWellPos[idx[2]].cmp);
            n1 = L1 + pi2;
            while (L21 - n1 > pi2) L21 -= pi;
            while (L21 - n1 < -pi2) n1 -= pi;
            L2 = (n1 - L21) * (1.0 * rand() / RAND_MAX) + L21;
            if (L2 < -pi) L2 += pi * 2;
            else if (L2 > pi) L2 -= pi * 2;

            // 交点
            if (fabs(L1) == pi2) {
                p01 = GPoint(_vWellPos[idx[1]].cmp, tan(L0) * (_vWellPos[idx[1]].cmp - _vWellPos[idx[0]].cmp) + _vWellPos[idx[0]].line);
                p12 = GPoint(_vWellPos[idx[1]].cmp, tan(L2) * (_vWellPos[idx[1]].cmp - _vWellPos[idx[2]].cmp) + _vWellPos[idx[2]].line);
            } else {
                if (fabs(L0) == pi2)
                    p01 = GPoint(_vWellPos[idx[0]].cmp, tan(L1) * (_vWellPos[idx[0]].cmp - _vWellPos[idx[1]].cmp) + _vWellPos[idx[1]].line);
                else {
                    double x = (tan(L0) * _vWellPos[idx[0]].cmp - tan(L1) * _vWellPos[idx[1]].cmp + _vWellPos[idx[1]].line - _vWellPos[idx[0]].line) / (tan(L0) - tan(L1));
                    p01 = GPoint(x, tan(L1) * (x - _vWellPos[idx[1]].cmp) + _vWellPos[idx[1]].line);
                }
                if (fabs(L2) == pi2)
                    p12 = GPoint(_vWellPos[idx[2]].cmp, tan(L1) * (_vWellPos[idx[2]].cmp - _vWellPos[idx[1]].cmp) + _vWellPos[idx[1]].line);
                else {
                    double x = (tan(L2) * _vWellPos[idx[2]].cmp - tan(L1) * _vWellPos[idx[1]].cmp + _vWellPos[idx[1]].line - _vWellPos[idx[2]].line) / (tan(L2) - tan(L1));
                    p12 = GPoint(x, tan(L1) * (x - _vWellPos[idx[1]].cmp) + _vWellPos[idx[1]].line);
                }
            }
            if (!_maxGridRange.IsValidLineCmpNo(etGridCrd(int(round(p01.GetY())), int(round(p01.GetX())))) ||
                !_maxGridRange.IsValidLineCmpNo(etGridCrd(int(round(p12.GetY())), int(round(p12.GetX()))))) {
                continue;
            }
            break;
        } while (true);

        // 端点
        std::vector<GPoint> vPoint;
        double x, y;
        x = _vWellPos[idx[0]].cmp > p01.GetX() ? _maxGridRange.GetCmpEnd() : _maxGridRange.GetCmpStart();
        y = _vWellPos[idx[0]].line > p01.GetY() ? _maxGridRange.GetLineEnd() : _maxGridRange.GetLineStart();
        if (fabs(L0) == M_PI_2)
            vPoint.push_back(GPoint(_vWellPos[idx[0]].cmp, y));
        else
            vPoint.push_back(GPoint(x, tan(L0) * (x - _vWellPos[idx[0]].cmp) + _vWellPos[idx[0]].line));
        vPoint.push_back(p01);
        vPoint.push_back(p12);
        x = _vWellPos[idx[2]].cmp > p12.GetX() ? _maxGridRange.GetCmpEnd() : _maxGridRange.GetCmpStart();
        y = _vWellPos[idx[2]].line > p12.GetY() ? _maxGridRange.GetLineEnd() : _maxGridRange.GetLineStart();
        if (fabs(L2) == M_PI_2)
            vPoint.push_back(GPoint(_vWellPos[idx[2]].cmp, y));
        else
            vPoint.push_back(GPoint(x, tan(L2) * (x - _vWellPos[idx[2]].cmp) + _vWellPos[idx[2]].line));
        vCrd = etDataTraverse::CalcGridPos(_maxGridRange, vPoint);

        vMask.resize(vCrd.size(), 0.0f);
        for (size_t i = 0; i < vCrd.size(); i++)
            if (_vWellMask.count(vCrd[i]))
                vMask[i] = _vWellMask[vCrd[i]];
        return true;
    }
};

// int main() {
//     srand((unsigned)time(0));
//     // 假设网格 0~99, 0~99, 有4口井
//     GridRange grid(0, 99, 0, 99);
//     std::vector<etGridCrd> wells = { {20,10}, {80,90}, {50,50}, {30,80} };
//     etWIInversionPredictJob job(wells, grid);

//     // 输出井数据(假定每井有10米深度，步长1米，属性为简单正弦示意)
//     int n_depth = 50;
//     for (size_t wi = 0; wi < wells.size(); ++wi) {
//         std::ofstream fout("well" + std::to_string(wi+1) + ".csv");
//         fout << "line,cmp,depth,curve\n";
//         for (int d = 0; d < n_depth; ++d) {
//             // 井直线，位置不变
//             double value = std::sin(0.2 * d + wi); // 示例测井曲线
//             fout << wells[wi].line << "," << wells[wi].cmp << "," << d << "," << value << "\n";
//         }
//         fout.close();
//     }

//     // 生成剖面
//     std::vector<etGridCrd> crd;
//     std::vector<float> mask;
//     if (job.GetWellLine(crd, mask)) {
//         std::ofstream fout("section.csv");
//         fout << "line,cmp\n";
//         for (size_t i = 0; i < crd.size(); ++i) {
//             fout << crd[i].line << "," << crd[i].cmp << "\n";
//         }
//         fout.close();
//         std::cout << "已保存剖面点到 section.csv\n";
//     } else {
//         std::cout << "剖面生成失败" << std::endl;
//     }

//     return 0;
// }

// #include <fstream>
// // ... 你的其他代码

int main() {
    srand((unsigned)time(0));
    GridRange grid(0, 99, 0, 99);
    std::vector<etGridCrd> wells = { {20,10}, {80,90}, {50,50}, {30,80} };
    etWIInversionPredictJob job(wells, grid);

    std::vector<etGridCrd> crd;
    std::vector<float> mask;
    if (job.GetWellLine(crd, mask)) {
        // 保存为csv
        std::ofstream fout("section.csv");
        fout << "line,cmp,mask\n";
        for (size_t i = 0; i < crd.size(); ++i) {
            fout << crd[i].line << "," << crd[i].cmp << "," << mask[i] << "\n";
        }
        fout.close();
        std::cout << "已保存剖面数据到 section.csv\n";
    } else {
        std::cout << "剖面生成失败" << std::endl;
    }
    return 0;
}