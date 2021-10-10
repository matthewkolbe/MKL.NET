﻿using System;

namespace MKLNET;

/// <summary>A median and quartile estimator.</summary>
public class QuartileEstimator
{
    /// <summary>The number of sample observations.</summary>
    public int N;
    int N1 = 2, N2 = 3, N3 = 4;
    /// <summary>The minimum or 0th percentile.</summary>
    public double Q0;
    /// <summary>The first, lower quartile, or 25th percentile.</summary>
    public double Q1;
    /// <summary>The second quartile, median, or 50th percentile.</summary>
    public double Q2;
    /// <summary>The third, upper quartile, or 75th percentile.</summary>
    public double Q3;
    /// <summary>The maximum or 100th percentile.</summary>
    public double Q4;
    /// <summary>The minimum or 0th percentile.</summary>
    public double Minimum => Q0;
    /// <summary>The first, lower quartile, or 25th percentile.</summary>
    public double LowerQuartile => Q1;
    /// <summary>The second quartile, median, or 50th percentile.</summary>
    public double Median => Q2;
    /// <summary>The third, upper quartile, or 75th percentile.</summary>
    public double UpperQuartile => Q3;
    /// <summary>The maximum or 100th percentile.</summary>
    public double Maximum => Q4;
    /// <summary>Add a sample observation.</summary>
    /// <param name="s">Sample observation value.</param>
    public void Add(double s)
    {
        if (++N > 5)
        {
            if (s < Q3)
            {
                N3++;
                if (s < Q2)
                {
                    N2++;
                    if (s < Q1)
                    {
                        N1++;
                        if (s < Q0) Q0 = s;
                    }
                }
            }
            else if (s > Q4) Q4 = s;

            s = 1 - N1 + (N - 1) * 0.25;
            if ((s >= 1.0 && N2 - N1 > 1) || (s <= -1.0 && 1 - N1 < -1))
            {
                int ds = Math.Sign(s);
                double q = Q1 + (double)ds / (N2 - 1) * ((N1 - 1 + ds) * (Q2 - Q1) / (N2 - N1) + (N2 - N1 - ds) * (Q1 - Q0) / (N1 - 1));
                q = Q0 < q && q < Q2 ? q :
                    ds == 1 ? Q1 + (Q2 - Q1) / (N2 - N1) :
                    Q1 - (Q0 - Q1) / (1 - N1);
                N1 += ds;
                Q1 = q;
            }
            s = 1 - N2 + (N - 1) * 0.50;
            if ((s >= 1.0 && N3 - N2 > 1) || (s <= -1.0 && N1 - N2 < -1))
            {
                int ds = Math.Sign(s);
                double q = Q2 + (double)ds / (N3 - N1) * ((N2 - N1 + ds) * (Q3 - Q2) / (N3 - N2) + (N3 - N2 - ds) * (Q2 - Q1) / (N2 - N1));
                q = Q1 < q && q < Q3 ? q :
                    ds == 1 ? Q2 + (Q3 - Q2) / (N3 - N2) :
                    Q2 - (Q1 - Q2) / (N1 - N2);
                N2 += ds;
                Q2 = q;
            }
            s = 1 - N3 + (N - 1) * 0.75;
            if ((s >= 1.0 && N - N3 > 1) || (s <= -1.0 && N2 - N3 < -1))
            {
                int ds = Math.Sign(s);
                double q = Q3 + (double)ds / (N - N2) * ((N3 - N2 + ds) * (Q4 - Q3) / (N - N3) + (N - N3 - ds) * (Q3 - Q2) / (N3 - N2));
                q = Q2 < q && q < Q4 ? q :
                    ds == 1 ? Q3 + (Q4 - Q3) / (N - N3) :
                    Q3 - (Q2 - Q3) / (N2 - N3);
                N3 += ds;
                Q3 = q;
            }
        }
        else if (N == 5)
        {
            if (s > Q4)
            {
                Q0 = Q1;
                Q1 = Q2;
                Q2 = Q3;
                Q3 = Q4;
                Q4 = s;
            }
            else if (s > Q3)
            {
                Q0 = Q1;
                Q1 = Q2;
                Q2 = Q3;
                Q3 = s;
            }
            else if (s > Q2)
            {
                Q0 = Q1;
                Q1 = Q2;
                Q2 = s;
            }
            else if (s > Q1)
            {
                Q0 = Q1;
                Q1 = s;
            }
            else Q0 = s;
        }
        else if (N == 4)
        {
            if (s < Q1)
            {
                Q4 = Q3;
                Q3 = Q2;
                Q2 = Q1;
                Q1 = s;
            }
            else if (s < Q2)
            {
                Q4 = Q3;
                Q3 = Q2;
                Q2 = s;
            }
            else if (s < Q3)
            {
                Q4 = Q3;
                Q3 = s;
            }
            else Q4 = s;
        }
        else if (N == 3)
        {
            if (s < Q1)
            {
                Q3 = Q2;
                Q2 = Q1;
                Q1 = s;
            }
            else if (s < Q2)
            {
                Q3 = Q2;
                Q2 = s;
            }
            else Q3 = s;
        }
        else if (N == 2)
        {
            if (s > Q2)
            {
                Q1 = Q2;
                Q2 = s;
            }
            else Q1 = s;
        }
        else Q2 = s;
    }
    /// <summary>Combine another QuartileEstimator.</summary>
    /// <param name="qe">QuartileEstimator</param>
    public void Add(QuartileEstimator qe)
    {
        N += qe.N;
        N1 += qe.N1;
        N2 += qe.N2;
        N3 += qe.N3;
        if (qe.Q0 < Q0) Q0 = qe.Q0;
        if (qe.Q4 > Q4) Q4 = qe.Q4;
        var w = (double)qe.N / N;
        Q1 += (qe.Q1 - Q1) * w;
        Q2 += (qe.Q2 - Q2) * w;
        Q3 += (qe.Q3 - Q3) * w;
    }
    /// <summary>Combine two QuartileEstimators.</summary>
    /// <param name="a">First QuartileEstimator</param>
    /// <param name="b">Second QuartileEstimator</param>
    public static QuartileEstimator operator +(QuartileEstimator a, QuartileEstimator b)
    {
        var w = (double)b.N / (a.N + b.N);
        return new QuartileEstimator
        {
            N = a.N + b.N,
            N1 = a.N1 + b.N1,
            N2 = a.N2 + b.N2,
            N3 = a.N3 + b.N3,
            Q0 = Math.Min(a.Q0, b.Q0),
            Q1 = a.Q1 + (b.Q1 - a.Q1) * w,
            Q2 = a.Q2 + (b.Q2 - a.Q2) * w,
            Q3 = a.Q3 + (b.Q3 - a.Q3) * w,
            Q4 = Math.Min(a.Q4, b.Q4),
        };
    }
}