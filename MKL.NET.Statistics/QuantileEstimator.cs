﻿// Copyright 2021 Anthony Lloyd
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

namespace MKLNET
{
    /// <summary>A quantile estimator.</summary>
    public class QuantileEstimator
    {
        readonly double p;
        /// <summary>The number of sample observations.</summary>
        public int N;
        /// <summary>
        /// 
        /// </summary>
        public int N0 = 1, N1 = 2, N2 = 3, N3 = 4;
        double Q0, Q1, Q3, Q4;
        /// <summary>The quantile estimate.</summary>
        public double Quantile;

        /// <summary>A quantile estimator.</summary>
        /// <param name="p">The quantile 0.0-1.0 to estimate.</param>
        public QuantileEstimator(double p) => this.p = p;

        /// <summary>Add a sample observation.</summary>
        /// <param name="s">Sample observation value.</param>
        public void Add(double s)
        {
            if (++N > 5)
            {
                if (s <= Q3)
                {
                    N3++;
                    if (s <= Quantile)
                    {
                        N2++;
                        if (s <= Q1)
                        {
                            N1++;
                            if (s <= Q0)
                            {
                                if (s == Q0)
                                {
                                    N0++;
                                }
                                else
                                {
                                    Q0 = s;
                                    N0 = 1;
                                }
                            }
                        }
                    }
                }
                else if (s > Q4) Q4 = s;

                s = (N - 1) * p * 0.5 + 1 - N1;
                if (s >= 1.0 && N2 - N1 > 1)
                {
                    var h1 = N2 - N1;
                    var delta1 = (Quantile - Q1) / h1;
                    var d1 = Derivative(N1 - N0, (Q1 - Q0) / (N1 - N0), h1, delta1);
                    var d2 = Derivative(h1, delta1, N3 - N2, (Q3 - Quantile) / (N3 - N2));
                    Q1 = HermiteInterpolationOne(Q1, d1, d2, h1, delta1);
                    N1++;
                }
                else if (s <= -1.0 && N1 - N0 > 1)
                {
                    var h0 = N1 - N0;
                    var delta0 = (Q1 - Q0) / h0;
                    var d0 = DerivativeEnd(h0, delta0, N2 - N1, (Quantile - Q1) / (N2 - N1));
                    var d1 = Derivative(h0, delta0, N2 - N1, (Quantile - Q1) / (N2 - N1));
                    Q1 = HermiteInterpolationOne(Q1, -d1, -d0, h0, -delta0);
                    N1--;
                }
                s = (N - 1) * p + 1 - N2;
                if (s >= 1.0 && N3 - N2 > 1)
                {
                    var h2 = N3 - N2;
                    var delta2 = (Q3 - Quantile) / h2;
                    var d2 = Derivative(N2 - N1, (Quantile - Q1) / (N2 - N1), h2, delta2);
                    var d3 = Derivative(h2, delta2, N - N3, (Q4 - Q3) / (N - N3));
                    Quantile = HermiteInterpolationOne(Quantile, d2, d3, h2, delta2);
                    N2++;
                }
                else if (s <= -1.0 && N2 - N1 > 1)
                {
                    var h1 = N2 - N1;
                    var delta1 = (Quantile - Q1) / h1;
                    var d1 = Derivative(N1 - N0, (Q1 - Q0) / (N1 - N0), h1, delta1);
                    var d2 = Derivative(h1, delta1, N3 - N2, (Q3 - Quantile) / (N3 - N2));
                    Quantile = HermiteInterpolationOne(Quantile, -d2, -d1, h1, -delta1);
                    N2--;
                }
                s = (N - 1) * (1 + p) * 0.5 + 1 - N3;
                if (s >= 1.0 && N - N3 > 1)
                {
                    var h3 = N - N3;
                    var delta3 = (Q4 - Q3) / h3;
                    var d3 = Derivative(N3 - N2, (Q3 - Quantile) / (N3 - N2), h3, delta3);
                    var d4 = DerivativeEnd(h3, delta3, N3 - N2, (Q3 - Quantile) / (N3 - N2));
                    Q3 = HermiteInterpolationOne(Q3, d3, d4, h3, delta3);
                    N3++;
                }
                else if (s <= -1.0 && N3 - N2 > 1)
                {
                    var h2 = N3 - N2;
                    var delta2 = (Q3 - Quantile) / h2;
                    var d2 = Derivative(N2 - N1, (Quantile - Q1) / (N2 - N1), h2, delta2);
                    var d3 = Derivative(h2, delta2, N - N3, (Q4 - Q3) / (N - N3));
                    Q3 = HermiteInterpolationOne(Q3, -d3, -d2, h2, -delta2);
                    N3--;
                }
            }
            else if (N == 5)
            {
                if (s > Q4)
                {
                    Q0 = Q1;
                    Q1 = Quantile;
                    Quantile = Q3;
                    Q3 = Q4;
                    Q4 = s;
                }
                else if (s > Q3)
                {
                    Q0 = Q1;
                    Q1 = Quantile;
                    Quantile = Q3;
                    Q3 = s;
                }
                else if (s > Quantile)
                {
                    Q0 = Q1;
                    Q1 = Quantile;
                    Quantile = s;
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
                    Q3 = Quantile;
                    Quantile = Q1;
                    Q1 = s;
                }
                else if (s < Quantile)
                {
                    Q4 = Q3;
                    Q3 = Quantile;
                    Quantile = s;
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
                    Q3 = Quantile;
                    Quantile = Q1;
                    Q1 = s;
                }
                else if (s < Quantile)
                {
                    Q3 = Quantile;
                    Quantile = s;
                }
                else Q3 = s;
            }
            else if (N == 2)
            {
                if (s > Quantile)
                {
                    Q1 = Quantile;
                    Quantile = s;
                }
                else Q1 = s;
            }
            else Quantile = s;
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        static double Derivative(int h1, double delta1, int h2, double delta2)
        {
            return (h1 + h2) * 3 * delta1 * delta2 / ((h1 * 2 + h2) * delta1 + (h2 * 2 + h1) * delta2);
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        static double DerivativeEnd(int h1, double delta1, int h2, double delta2)
        {
            double d = (delta1 - delta2) * h1 / (h1 + h2) + delta1;
            return d < 0.0 ? 0.0
                 : d > 3 * delta1 ? 3 * delta1
                 : d;
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        static double HermiteInterpolationOne(double y1, double d1, double d2, int h1, double delta1)
        {
            return ((d1 + d2 - delta1 * 2) / h1 + delta1 * 3 - d1 * 2 - d2) / h1 + y1 + d1;
        }

        /// <summary>Combine another QuartileEstimator.</summary>
        /// <param name="qe">QuartileEstimator</param>
        public void Add(QuantileEstimator qe)
        {
            N += qe.N;
            N1 += qe.N1;
            N2 += qe.N2;
            N3 += qe.N3;
            if (qe.Q0 < Q0) Q0 = qe.Q0;
            if (qe.Q4 > Q4) Q4 = qe.Q4;
            var w = (double)qe.N / N;
            Q1 += (qe.Q1 - Q1) * w;
            Quantile += (qe.Quantile - Quantile) * w;
            Q3 += (qe.Q3 - Q3) * w;
        }

        /// <summary>Combine two QuartileEstimators.</summary>
        /// <param name="a">First QuartileEstimator</param>
        /// <param name="b">Second QuartileEstimator</param>
        public static QuantileEstimator operator +(QuantileEstimator a, QuantileEstimator b)
        {
            var w = (double)b.N / (a.N + b.N);
            return new QuantileEstimator(a.p)
            {
                N = a.N + b.N,
                N1 = a.N1 + b.N1,
                N2 = a.N2 + b.N2,
                N3 = a.N3 + b.N3,
                Q0 = a.Q0 < b.Q0 ? a.Q0 : b.Q0,
                Q1 = a.Q1 + (b.Q1 - a.Q1) * w,
                Quantile = a.Quantile + (b.Quantile - a.Quantile) * w,
                Q3 = a.Q3 + (b.Q3 - a.Q3) * w,
                Q4 = a.Q4 > b.Q4 ? a.Q4 : b.Q4,
            };
        }
    }
}