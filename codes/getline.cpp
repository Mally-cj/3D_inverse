
//利用井口坐标生成掩码缓存
QMap<etGridCrd, float> _vWellMask;
int wellRange = 15;
for (int i = 0; i < _vWellPos.size(); i++)
{
    for (int iline = -wellRange; iline <= wellRange; iline++)
        for (int icmp = -wellRange; icmp <= wellRange; icmp++)
        {
            etGridCrd crd(_vWellPos[i].line + iline, _vWellPos[i].cmp + icmp);
            if (!_vWellMask.contains(crd))
                _vWellMask[crd] = exp(-(iline * iline + icmp * icmp) / 50.0);
            else
            {
                float weight = exp(-(iline * iline + icmp * icmp) / 50.0);
                if (_vWellMask[crd] < weight)
                    _vWellMask[crd] = weight;
            }
        }
}

//过随机三井连井线
bool etWIInversionPredictJob::GetWellLine(QVector<etGridCrd>&vCrd, QVector<float>&vMask)
{
    QVector<int> idx;
    for (int i = 0; i < 3; i++)
    {
        int tmp = rand() % _vWellPos.size();
        while (idx.contains(tmp))
            tmp = rand() % _vWellPos.size();
        idx << tmp;
    }

    GPoint p01, p12;
    float L0, L1, L2;
    do {
        //长边对大角，取最大角为中间井
        float d01 = (_vWellPos[idx[0]].cmp - _vWellPos[idx[1]].cmp) * (_vWellPos[idx[0]].cmp - _vWellPos[idx[1]].cmp) + (_vWellPos[idx[0]].line - _vWellPos[idx[1]].line) * (_vWellPos[idx[0]].line - _vWellPos[idx[1]].line);
        float d12 = (_vWellPos[idx[1]].cmp - _vWellPos[idx[2]].cmp) * (_vWellPos[idx[1]].cmp - _vWellPos[idx[2]].cmp) + (_vWellPos[idx[1]].line - _vWellPos[idx[2]].line) * (_vWellPos[idx[1]].line - _vWellPos[idx[2]].line);
        float d20 = (_vWellPos[idx[2]].cmp - _vWellPos[idx[0]].cmp) * (_vWellPos[idx[2]].cmp - _vWellPos[idx[0]].cmp) + (_vWellPos[idx[2]].line - _vWellPos[idx[0]].line) * (_vWellPos[idx[2]].line - _vWellPos[idx[0]].line);
        if (d01 <= d20 && d20 <= d12)
            idx = QVector<int>() << idx[1] << idx[0] << idx[2];
        else if (d12 <= d01 && d01 <= d20)
            idx = QVector<int>() << idx[2] << idx[1] << idx[0];
        else if (d12 <= d20 && d20 <= d01)
            idx = QVector<int>() << idx[1] << idx[2] << idx[0];
        else if (d20 <= d01 && d01 <= d12)
            idx = QVector<int>() << idx[2] << idx[0] << idx[1];
        else if (d20 <= d12 && d12 <= d01)
            idx = QVector<int>() << idx[0] << idx[2] << idx[1];

        //最大角两边法向量
        float n10 = atan2(_vWellPos[idx[0]].line - _vWellPos[idx[1]].line, _vWellPos[idx[0]].cmp - _vWellPos[idx[1]].cmp) + M_PI / 2;
        float n12 = atan2(_vWellPos[idx[2]].line - _vWellPos[idx[1]].line, _vWellPos[idx[2]].cmp - _vWellPos[idx[1]].cmp) + M_PI / 2;
        //取正负pi区间
        if (n10 > M_PI)
            n10 -= M_PI * 2;
        if (n12 > M_PI)
            n12 -= M_PI * 2;
        if (n10 - n12 > M_PI)
            n10 -= M_PI * 2;
        else if (n10 - n12 < -M_PI)
            n12 -= M_PI * 2;
        //1边向量
        L1 = (n12 - n10) * (1.0 * rand() / RAND_MAX) + n10;
        if (L1 < -M_PI)
            L1 += M_PI * 2;
        else if (L1 > M_PI)
            L1 -= M_PI * 2;

        //01井连线
        float L01 = atan2(_vWellPos[idx[1]].line - _vWellPos[idx[0]].line, _vWellPos[idx[1]].cmp - _vWellPos[idx[0]].cmp);
        //1边法向量
        float n1 = L1 + M_PI / 2;
        //取锐角区间
        while (L01 - n1 > M_PI_2)
            L01 -= M_PI;
        while (L01 - n1 < -M_PI_2)
            n1 -= M_PI;
        //0边向量
        L0 = (n1 - L01) * (1.0 * rand() / RAND_MAX) + L01;
        if (L0 < -M_PI)
            L0 += M_PI * 2;
        else if (L0 > M_PI)
            L0 -= M_PI * 2;

        //21井连线
        float L21 = atan2(_vWellPos[idx[1]].line - _vWellPos[idx[2]].line, _vWellPos[idx[1]].cmp - _vWellPos[idx[2]].cmp);
        //1边法向量
        n1 = L1 + M_PI / 2;
        //取锐角区间
        while (L21 - n1 > M_PI_2)
            L21 -= M_PI;
        while (L21 - n1 < -M_PI_2)
            n1 -= M_PI;
        //2边向量
        L2 = (n1 - L21) * (1.0 * rand() / RAND_MAX) + L21;
        if (L2 < -M_PI)
            L2 += M_PI * 2;
        else if (L2 > M_PI)
            L2 -= M_PI * 2;

        //交点
        if (abs(L1) == M_PI_2)
        {
            p01 = GPoint(_vWellPos[idx[1]].cmp, tan(L0) * (_vWellPos[idx[1]].cmp - _vWellPos[idx[0]].cmp) + _vWellPos[idx[0]].line);
            p12 = GPoint(_vWellPos[idx[1]].cmp, tan(L2) * (_vWellPos[idx[1]].cmp - _vWellPos[idx[2]].cmp) + _vWellPos[idx[2]].line);
        }
        else
        {
            if (abs(L0) == M_PI_2)
                p01 = GPoint(_vWellPos[idx[0]].cmp, tan(L1) * (_vWellPos[idx[0]].cmp - _vWellPos[idx[1]].cmp) + _vWellPos[idx[1]].line);
            else
            {
                float x = (tan(L0) * _vWellPos[idx[0]].cmp - tan(L1) * _vWellPos[idx[1]].cmp + _vWellPos[idx[1]].line - _vWellPos[idx[0]].line) / (tan(L0) - tan(L1));
                p01 = GPoint(x, tan(L1) * (x - _vWellPos[idx[1]].cmp) + _vWellPos[idx[1]].line);
            }
            if (abs(L2) == M_PI_2)
                p12 = GPoint(_vWellPos[idx[2]].cmp, tan(L1) * (_vWellPos[idx[2]].cmp - _vWellPos[idx[1]].cmp) + _vWellPos[idx[1]].line);
            else
            {
                float x = (tan(L2) * _vWellPos[idx[2]].cmp - tan(L1) * _vWellPos[idx[1]].cmp + _vWellPos[idx[1]].line - _vWellPos[idx[2]].line) / (tan(L2) - tan(L1));
                p12 = GPoint(x, tan(L1) * (x - _vWellPos[idx[1]].cmp) + _vWellPos[idx[1]].line);
            }
        }
    } while (!_maxGridRange.IsValidLineCmpNo(etGridCrd() = p01) || !_maxGridRange.IsValidLineCmpNo(etGridCrd() = p12));//确保交点在计算范围内


    //利用端点交点坐标生成连井线
    QVector<GPoint> vPoint;
    double x = _vWellPos[idx[0]].cmp > p01.GetX() ? _maxGridRange.GetCmpEnd() : _maxGridRange.GetCmpStart();
    double y = _vWellPos[idx[0]].line > p01.GetY() ? _maxGridRange.GetLineEnd() : _maxGridRange.GetLineStart();
    if (abs(L0) == M_PI_2)
        vPoint << GPoint(_vWellPos[idx[0]].cmp, y);
    else
        vPoint << GPoint(x, tan(L0) * (x - _vWellPos[idx[0]].cmp) + _vWellPos[idx[0]].line);
    vPoint << p01 << p12;
    x = _vWellPos[idx[2]].cmp > p12.GetX() ? _maxGridRange.GetCmpEnd() : _maxGridRange.GetCmpStart();
    y = _vWellPos[idx[2]].line > p12.GetY() ? _maxGridRange.GetLineEnd() : _maxGridRange.GetLineStart();
    if (abs(L2) == M_PI_2)
        vPoint << GPoint(_vWellPos[idx[2]].cmp, y);
    else
        vPoint << GPoint(x, tan(L2) * (x - _vWellPos[idx[2]].cmp) + _vWellPos[idx[2]].line);
    vCrd = etDataTraverse::CalcGridPos(_maxGridRange, vPoint);

//得到对应掩码
    vMask.resize(vCrd.size());
    vMask.fill(0.0f);
    for (int i = 0; i < vCrd.size(); i++)
    {
        if (_vWellMask.contains(vCrd[i]))
            vMask[i] = _vWellMask[vCrd[i]];
    }

    return true;
}