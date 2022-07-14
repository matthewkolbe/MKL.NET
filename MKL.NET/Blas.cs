// Copyright 2022 Anthony Lloyd
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

namespace MKLNET;

using System;
using System.Security;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;

#pragma warning disable IDE1006 // Naming Styles

[SuppressUnmanagedCodeSecurity]
public static class Blas
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float dot(int N, float[] X, int iniX, int incX, float[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                return Native.cblas_sdot(N, xp, incX, yp, incY);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float dot(float[] X, float[] Y)
        => dot(X.Length, X, 0, 1, Y, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float dot(Span<float> X, Span<float> Y)
    {
        unsafe 
        {
            fixed (float* xp = X) fixed(float* yp = Y)
                return Native.cblas_sdot(X.Length, xp, 1, yp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float dot(ref Span<float> X, ref Span<float> Y)
    {
        unsafe 
        {
            fixed (float* xp = X) fixed(float* yp = Y)
                return Native.cblas_sdot(X.Length, xp, 1, yp, 1);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float doti(int N, float[] X, int[] indx, float[] Y)
        => Native.cblas_sdoti(N, X, indx, Y);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double dot(int N, double[] X, int iniX, int incX, double[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                return Native.cblas_ddot(N, xp, incX, yp, incY);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double dot(double[] X, double[] Y)
        => dot(X.Length, X, 0, 1, Y, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double dot(Span<double> X, Span<double> Y)
    {
        unsafe
        {
            fixed (double* xp = X) fixed(double* yp = Y)
                return Native.cblas_ddot(X.Length, xp, 1, yp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double dot(ref Span<double> X, ref Span<double> Y)
    {
        unsafe 
        {
            fixed (double* xp = X) fixed(double* yp = Y)
                return Native.cblas_ddot(X.Length, xp, 1, yp, 1);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double doti(int N, double[] X, int[] indx, double[] Y)
        => Native.cblas_ddoti(N, X, indx, Y);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double sdot(int N, float[] X, int iniX, int incX, float[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                return Native.cblas_dsdot(N, xp, incX, yp, incY);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double sdot(float[] X, float[] Y)
        => sdot(X.Length, X, 0, 1, Y, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double sdot(Span<float> X, Span<float> Y)
    {
        unsafe
        {
            fixed (float* xp = X) fixed (float* yp = Y)
                return Native.cblas_dsdot(X.Length, xp, 1, yp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double sdot(ref Span<float> X, ref Span<float> Y)
    {
        unsafe
        {
            fixed (float* xp = X) fixed (float* yp = Y)
                return Native.cblas_dsdot(X.Length, xp, 1, yp, 1);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float sdot(int N, float sb, float[] X, int iniX, int incX, float[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                return Native.cblas_sdsdot(N, sb, xp, incX, yp, incY);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float sdot(float sb, float[] X, float[] Y)
        => sdot(X.Length, sb, X, 0, 1, Y, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float sdot(float sb, Span<float> X, Span<float> Y)
    {
        unsafe
        {
            fixed (float* xp = X) fixed (float* yp = Y)
                return Native.cblas_sdsdot(X.Length, sb, xp, 1, yp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float sdot(float sb, ref Span<float> X, ref Span<float> Y)
    {
        unsafe
        {
            fixed (float* xp = X) fixed (float* yp = Y)
                return Native.cblas_sdsdot(X.Length, sb, xp, 1, yp, 1);
        }
    }
#endif


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float nrm2(int N, float[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX])
                return Native.cblas_snrm2(N, xp, incX);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float nrm2(float[] X)
        => nrm2(X.Length, X, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float nrm2(Span<float> X)
    {
        unsafe
        {
            fixed (float* xp = X)
                return Native.cblas_snrm2(X.Length, xp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float nrm2(ref Span<float> X)
    {
        unsafe
        {
            fixed (float* xp = X)
                return Native.cblas_snrm2(X.Length, xp, 1);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float asum(int N, float[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX])
                return Native.cblas_sasum(N, xp, incX);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float asum(float[] X)
        => asum(X.Length, X, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float asum(Span<float> X)
    {
        unsafe
        {
            fixed (float* xp = X)
                return Native.cblas_sasum(X.Length, xp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float asum(ref Span<float> X)
    {
        unsafe
        {
            fixed (float* xp = X)
                return Native.cblas_sasum(X.Length, xp, 1);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double nrm2(int N, double[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX])
                return Native.cblas_dnrm2(N, xp, incX);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double nrm2(double[] X)
        => nrm2(X.Length, X, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double nrm2(Span<double> X)
    {
        unsafe
        {
            fixed (double* xp = X)
                return Native.cblas_dnrm2(X.Length, xp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double nrm2(ref Span<double> X)
    {
        unsafe
        {
            fixed (double* xp = X)
                return Native.cblas_dnrm2(X.Length, xp, 1);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double asum(int N, double[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX])
                return Native.cblas_dasum(N, xp, incX);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double asum(double[] X)
        => asum(X.Length, X, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double asum(Span<double> X)
    {
        unsafe
        {
            fixed (double* xp = X)
                return Native.cblas_dasum(X.Length, xp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double asum(ref Span<double> X)
    {
        unsafe
        {
            fixed (double* xp = X)
                return Native.cblas_dasum(X.Length, xp, 1);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int iamax(int N, float[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX])
                return Native.cblas_isamax(N, xp, incX);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int iamax(float[] X)
        => iamax(X.Length, X, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int iamax(Span<float> X)
    {
        unsafe
        {
            fixed (float* xp = X)
                return Native.cblas_isamax(X.Length, xp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int iamax(ref Span<float> X)
    {
        unsafe
        {
            fixed (float* xp = X)
                return Native.cblas_isamax(X.Length, xp, 1);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int iamax(int N, double[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX])
                return Native.cblas_idamax(N, xp, incX);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int iamax(double[] X)
        => iamax(X.Length, X, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int iamax(Span<double> X)
    {
        unsafe
        {
            fixed (double* xp = X)
                return Native.cblas_idamax(X.Length, xp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int iamax(ref Span<double> X)
    {
        unsafe
        {
            fixed (double* xp = X)
                return Native.cblas_idamax(X.Length, xp, 1);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int iamin(int N, float[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX])
                return Native.cblas_isamin(N, xp, incX);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int iamin(float[] X)
        => iamin(X.Length, X, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int iamin(Span<float> X)
    {
        unsafe
        {
            fixed (float* xp = X)
                return Native.cblas_isamin(X.Length, xp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int iamin(ref Span<float> X)
    {
        unsafe
        {
            fixed (float* xp = X)
                return Native.cblas_isamin(X.Length, xp, 1);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int iamin(int N, double[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX])
                return Native.cblas_idamin(N, xp, incX);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int iamin(double[] X)
        => iamin(X.Length, X, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int iamin(Span<double> X)
    {
        unsafe
        {
            fixed (double* xp = X)
                return Native.cblas_idamin(X.Length, xp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int iamin(ref Span<double> X)
    {
        unsafe
        {
            fixed (double* xp = X)
                return Native.cblas_idamin(X.Length, xp, 1);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void swap(int N, float[] X, int iniX, int incX, float[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_sswap(N, xp, incX, yp, incY);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void swap(float[] X, float[] Y)
        => swap(X.Length, X, 0, 1, Y, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void swap(Span<float> X, Span<float> Y)
    {
        unsafe
        {
            fixed (float* xp = X) fixed (float* yp = Y)
                Native.cblas_sswap(X.Length, xp, 1, yp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void swap(ref Span<float> X, ref Span<float> Y)
    {
        unsafe
        {
            fixed (float* xp = X) fixed (float* yp = Y)
                Native.cblas_sswap(X.Length, xp, 1, yp, 1);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void copy(int N, float[] X, int iniX, int incX, float[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_scopy(N, xp, incX, yp, incY);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void copy(float[] X, float[] Y)
        => copy(X.Length, X, 0, 1, Y, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void copy(Span<float> X, Span<float> Y)
    {
        unsafe
        {
            fixed (float* xp = X) fixed (float* yp = Y)
                Native.cblas_scopy(X.Length, xp, 1, yp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void copy(ref Span<float> X, ref Span<float> Y)
    {
        unsafe
        {
            fixed (float* xp = X) fixed (float* yp = Y)
                Native.cblas_scopy(X.Length, xp, 1, yp, 1);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void axpy(int N, float a, float[] X, int iniX, int incX, float[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_saxpy(N, a, xp, incX, yp, incY);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void axpy(float a, float[] X, float[] Y)
        => axpy(X.Length, a, X, 0, 1, Y, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void axpy(float a, Span<float> X, Span<float> Y)
    {
        unsafe
        {
            fixed (float* xp = X) fixed (float* yp = Y)
                Native.cblas_saxpy(X.Length, a, xp, 1, yp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void axpy(float a, ref Span<float> X, ref Span<float> Y)
    {
        unsafe
        {
            fixed (float* xp = X) fixed (float* yp = Y)
                Native.cblas_saxpy(X.Length, a, xp, 1, yp, 1);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void axpby(int N, float alpha, float[] X, int iniX, int incX, float beta, float[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_saxpby(N, alpha, xp, incX, beta, yp, incY);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void axpby(float alpha, float[] X, float beta, float[] Y)
        => axpby(X.Length, alpha, X, 0, 1, beta, Y, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void axpby(float alpha, Span<float> X, float beta, Span<float> Y)
    {
        unsafe
        {
            fixed (float* xp = X) fixed (float* yp = Y)
                Native.cblas_saxpby(X.Length, alpha, xp, 1, beta, yp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void axpby(float alpha, ref Span<float> X, float beta, ref Span<float> Y)
    {
        unsafe
        {
            fixed (float* xp = X) fixed (float* yp = Y)
                Native.cblas_saxpby(X.Length, alpha, xp, 1, beta, yp, 1);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void axpyi(int N, float alpha, float[] X, int[] indx, float[] Y)
        => Native.cblas_saxpyi(N, alpha, X, indx, Y);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gthr(int N, float[] Y, float[] X, int[] indx)
        => Native.cblas_sgthr(N, Y, X, indx);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gthrz(int N, float[] Y, float[] X, int[] indx)
        => Native.cblas_sgthrz(N, Y, X, indx);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void sctr(int N, float[] X, int[] indx, float[] Y)
        => Native.cblas_ssctr(N, X, indx, Y);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rotg(ref float a, ref float b, ref float c, ref float s)
        => Native.cblas_srotg(ref a, ref b, ref c, ref s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void swap(int N, double[] X, int iniX, int incX, double[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_dswap(N, xp, incX, yp, incY);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void swap(double[] X, double[] Y)
        => swap(X.Length, X, 0, 1, Y, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void swap(Span<double> X, Span<double> Y)
    {
        unsafe
        {
            fixed (double* xp = X) fixed (double* yp = Y)
                Native.cblas_dswap(X.Length, xp, 1, yp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void swap(ref Span<double> X, ref Span<double> Y)
    {
        unsafe
        {
            fixed (double* xp = X) fixed (double* yp = Y)
                Native.cblas_dswap(X.Length, xp, 1, yp, 1);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void copy(int N, double[] X, int iniX, int incX, double[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_dcopy(N, xp, incX, yp, incY);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void copy(double[] X, double[] Y)
        => copy(X.Length, X, 0, 1, Y, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void copy(Span<double> X, Span<double> Y)
    {
        unsafe
        {
            fixed (double* xp = X) fixed (double* yp = Y)
                Native.cblas_dcopy(X.Length, xp, 1, yp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void copy(ref Span<double> X, ref Span<double> Y)
    {
        unsafe
        {
            fixed (double* xp = X) fixed (double* yp = Y)
                Native.cblas_dcopy(X.Length, xp, 1, yp, 1);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void axpy(int N, double a, double[] X, int iniX, int incX, double[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_daxpy(N, a, xp, incX, yp, incY);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void axpy(double a, double[] X, double[] Y)
        => axpy(X.Length, a, X, 0, 1, Y, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void axpy(double a, Span<double> X, Span<double> Y)
    {
        unsafe
        {
            fixed (double* xp = X) fixed (double* yp = Y)
                Native.cblas_daxpy(X.Length, a, xp, 1, yp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void axpy(double a, ref Span<double> X, ref Span<double> Y)
    {
        unsafe
        {
            fixed (double* xp = X) fixed (double* yp = Y)
                Native.cblas_daxpy(X.Length, a, xp, 1, yp, 1);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void axpby(int N, double alpha, double[] X, int iniX, int incX, double beta, double[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_daxpby(N, alpha, xp, incX, beta, yp, incY);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void axpby(double alpha, double[] X, double beta, double[] Y)
        => axpby(X.Length, alpha, X, 0, 1, beta, Y, 0, 1);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void axpby(double alpha, Span<double> X, double beta, Span<double> Y)
    {
        unsafe
        {
            fixed (double* xp = X) fixed (double* yp = Y)
                Native.cblas_daxpby(X.Length, alpha, xp, 1, beta, yp, 1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void axpby(double alpha, ref Span<double> X, double beta, ref Span<double> Y)
    {
        unsafe
        {
            fixed (double* xp = X) fixed (double* yp = Y)
                Native.cblas_daxpby(X.Length, alpha, xp, 1, beta, yp, 1);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void axpyi(int N, double alpha, double[] X, int[] indx, double[] Y)
        => Native.cblas_daxpyi(N, alpha, X, indx, Y);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gthr(int N, double[] Y, double[] X, int[] indx)
        => Native.cblas_dgthr(N, Y, X, indx);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gthrz(int N, double[] Y, double[] X, int[] indx)
        => Native.cblas_dgthrz(N, Y, X, indx);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void sctr(int N, double[] X, int[] indx, double[] Y)
        => Native.cblas_dsctr(N, X, indx, Y);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rotg(ref double a, ref double b, ref double c, ref double s)
        => Native.cblas_drotg(ref a, ref b, ref c, ref s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rotmg(ref float d1, ref float d2, ref float x1, float y1, float[] param)
    {
        unsafe
        {
            fixed (float* pp = &param[0])
                Native.cblas_srotmg(ref d1, ref d2, ref x1, y1, pp);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rot(int N, float[] X, int iniX, int incX, float[] Y, int iniY, int incY, float c, float s)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_srot(N, xp, incX, yp, incY, c, s);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rot(float[] X, float[] Y, float c, float s)
        => rot(X.Length, X, 0, 1, Y, 0, 1, c, s);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rot(Span<float> X, Span<float> Y, float c, float s)
    {
        unsafe
        {
            fixed (float* xp = X) fixed (float* yp = Y)
                Native.cblas_srot(X.Length, xp, 1, yp, 1, c, s);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rot(ref Span<float> X, ref Span<float> Y, float c, float s)
    {
        unsafe
        {
            fixed (float* xp = X) fixed (float* yp = Y)
                Native.cblas_srot(X.Length, xp, 1, yp, 1, c, s);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void roti(int N, float[] X, int[] indx, float[] Y, float c, float s)
        => Native.cblas_sroti(N, X, indx, Y, c, s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rotm(int N, float[] X, int iniX, int incX, float[] Y, int iniY, int incY, float[] param)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY]) fixed (float* pp = &param[0])
                Native.cblas_srotm(N, xp, incX, yp, incY, pp);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rotm(float[] X, float[] Y, float[] param)
        => rotm(X.Length, X, 0, 1, Y, 0, 1, param);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rotm(Span<float> X, Span<float> Y, Span<float> param)
    {
        unsafe
        {
            fixed (float* xp = X) fixed (float* yp = Y) fixed (float* pp = param)
                Native.cblas_srotm(X.Length, xp, 1, yp, 1, pp);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rotm(ref Span<float> X, ref Span<float> Y, ref Span<float> param)
    {
        unsafe
        {
            fixed (float* xp = X) fixed (float* yp = Y) fixed (float* pp = param)
                Native.cblas_srotm(X.Length, xp, 1, yp, 1, pp);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rotmg(ref double d1, ref double d2, ref double x1, double y1, double[] param)
    {
        unsafe
        {
            fixed (double* pp = &param[0])
                Native.cblas_drotmg(ref d1, ref d2, ref x1, y1, pp);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rot(int N, double[] X, int iniX, int incX, double[] Y, int iniY, int incY, double c, double s)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_drot(N, xp, incX, yp, incY, c, s);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rot(double[] X, double[] Y, double c, double s)
        => rot(X.Length, X, 0, 1, Y, 0, 1, c, s);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rot(Span<double> X, Span<double> Y, double c, double s)
    {
        unsafe
        {
            fixed (double* xp = X) fixed (double* yp = Y)
                Native.cblas_drot(X.Length, xp, 1, yp, 1, c, s);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rot(ref Span<double> X, ref Span<double> Y, ref double c, ref double s)
    {
        unsafe
        {
            fixed (double* xp = X) fixed (double* yp = Y)
                Native.cblas_drot(X.Length, xp, 1, yp, 1, c, s);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rotm(int N, double[] X, int iniX, int incX, double[] Y, int iniY, int incY, double[] param)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY]) fixed (double* pp = &param[0])
                Native.cblas_drotm(N, xp, incX, yp, incY, pp);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rotm(double[] X, double[] Y, double[] param)
        => rotm(X.Length, X, 0, 1, Y, 0, 1, param);

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rotm(Span<double> X, Span<double> Y, Span<double> param)
    {
        unsafe
        {
            fixed (double* xp = X) fixed (double* yp = Y) fixed (double* pp = param)
                Native.cblas_drotm(X.Length, xp, 1, yp, 1, pp);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void rotm(ref Span<double> X, ref Span<double> Y, ref Span<double> param)
    {
        unsafe
        {
            fixed (double* xp = X) fixed (double* yp = Y) fixed (double* pp = param)
                Native.cblas_drotm(X.Length, xp, 1, yp, 1, pp);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void roti(int N, double[] X, int[] indx, double[] Y, double c, double s)
        => Native.cblas_droti(N, X, indx, Y, c, s);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void scal(int N, float a, float[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX])
                Native.cblas_sscal(N, a, xp, incX);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void scal(float a, float[] X)
        => scal(X.Length, a, X, 0, 1);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void scal(int N, double a, double[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX])
                Native.cblas_dscal(N, a, xp, incX);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void scal(double a, double[] X)
        => scal(X.Length, a, X, 0, 1);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gemv(Layout Layout,
        Trans TransA, int M, int N,
        float alpha, float[] A, int lda,
        float[] X, int iniX, int incX, float beta,
        float[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_sgemv(Layout, TransA, M, N, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gemv(Layout Layout,
        Trans TransA, int M, int N,
        float alpha, Span<float> A, int lda,
        Span<float> X, int iniX, int incX, float beta, 
        Span<float> Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_sgemv(Layout, TransA, M, N, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gemv(ref Layout Layout,
        ref Trans TransA, ref int M, ref int N,
        ref float alpha, ref Span<float> A, ref int lda,
        ref Span<float> X, ref int iniX, ref int incX, ref float beta,
        ref Span<float> Y, ref int iniY, ref int incY)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_sgemv(Layout, TransA, M, N, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gbmv(Layout Layout,
        Trans TransA, int M, int N,
        int KL, int KU, float alpha,
        float[] A, int lda, float[] X, int iniX,
        int incX, float beta, float[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_sgbmv(Layout, TransA, M, N, KL, KU, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gbmv(Layout Layout,
        Trans TransA, int M, int N,
        int KL, int KU, float alpha,
        Span<float> A, int lda, Span<float> X, int iniX,
        int incX, float beta, Span<float> Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_sgbmv(Layout, TransA, M, N, KL, KU, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gbmv(ref Layout Layout,
        ref Trans TransA, ref int M, ref int N,
        ref int KL, ref int KU, ref float alpha,
        ref Span<float> A, ref int lda, ref Span<float> X, ref int iniX,
        ref int incX, ref float beta, ref Span<float> Y, ref int iniY, ref int incY)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_sgbmv(Layout, TransA, M, N, KL, KU, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void trmv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, float[] A, int lda,
        float[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX])
                Native.cblas_strmv(Layout, UPLO, TransA, Diag, N, ap, lda, xp, incX);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void trmv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag, int N, Span<float> A, 
        int lda, Span<float> X, int iniX, int incX)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX])
                Native.cblas_strmv(Layout, UPLO, TransA, Diag, N, ap, lda, xp, incX);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void trmv(ref Layout Layout, ref UpLo UPLO,
        ref Trans TransA, ref Diag Diag, ref int N, ref Span<float> A, 
        ref int lda, ref Span<float> X, ref int iniX, ref int incX)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX])
                Native.cblas_strmv(Layout, UPLO, TransA, Diag, N, ap, lda, xp, incX);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tbmv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, int K, float[] A, int lda,
        float[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX])
                Native.cblas_stbmv(Layout, UPLO, TransA, Diag, N, K, ap, lda, xp, incX);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tbmv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag, int N, int K, 
        Span<float> A, int lda, Span<float> X, int iniX, int incX)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX])
                Native.cblas_stbmv(Layout, UPLO, TransA, Diag, N, K, ap, lda, xp, incX);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tbmv(ref Layout Layout, ref UpLo UPLO,
        ref Trans TransA, ref Diag Diag, ref int N, ref int K, 
        ref Span<float> A, ref int lda, ref Span<float> X, ref int iniX, ref int incX)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX])
                Native.cblas_stbmv(Layout, UPLO, TransA, Diag, N, K, ap, lda, xp, incX);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tpmv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, float[] Ap, float[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (float* ap = &Ap[0]) fixed (float* xp = &X[iniX])
                Native.cblas_stpmv(Layout, UPLO, TransA, Diag, N, ap, xp, incX);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tpmv(Layout Layout, UpLo UPLO, Trans TransA, Diag Diag, int N,
        Span<float> Ap, Span<float> X, int iniX, int incX)
    {
        unsafe
        {
            fixed (float* ap = &Ap[0]) fixed (float* xp = &X[iniX])
                Native.cblas_stpmv(Layout, UPLO, TransA, Diag, N, ap, xp, incX);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tpmv(ref Layout Layout, ref UpLo UPLO, ref Trans TransA, 
        ref Diag Diag, ref int N, ref Span<float> Ap, ref Span<float> X, ref int iniX, ref int incX)
    {
        unsafe
        {
            fixed (float* ap = &Ap[0]) fixed (float* xp = &X[iniX])
                Native.cblas_stpmv(Layout, UPLO, TransA, Diag, N, ap, xp, incX);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void trsv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, float[] A, int lda, float[] X,
        int iniX, int incX)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX])
                Native.cblas_strsv(Layout, UPLO, TransA, Diag, N, ap, lda, xp, incX);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void trsv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag, int N, Span<float> A, 
        int lda, Span<float> X,
        int iniX, int incX)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX])
                Native.cblas_strsv(Layout, UPLO, TransA, Diag, N, ap, lda, xp, incX);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void trsv(ref Layout Layout, ref UpLo UPLO,
        ref Trans TransA, ref Diag Diag, ref int N, ref Span<float> A, 
        ref int lda, ref Span<float> X,
        int iniX, int incX)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX])
                Native.cblas_strsv(Layout, UPLO, TransA, Diag, N, ap, lda, xp, incX);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tbsv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, int K, float[] A, int lda,
        float[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX])
                Native.cblas_stbsv(Layout, UPLO, TransA, Diag, N, K, ap, lda, xp, incX);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tbsv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, int K, Span<float> A, int lda,
        Span<float> X, int iniX, int incX)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX])
                Native.cblas_stbsv(Layout, UPLO, TransA, Diag, N, K, ap, lda, xp, incX);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tbsv(ref Layout Layout, ref UpLo UPLO,
        ref Trans TransA, ref Diag Diag,
        ref int N, ref int K, ref Span<float> A, ref int lda,
        ref Span<float> X, ref int iniX, ref int incX)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX])
                Native.cblas_stbsv(Layout, UPLO, TransA, Diag, N, K, ap, lda, xp, incX);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tpsv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, float[] Ap, float[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (float* ap = &Ap[0]) fixed (float* xp = &X[iniX])
                Native.cblas_stpsv(Layout, UPLO, TransA, Diag, N, ap, xp, incX);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tpsv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, Span<float> Ap, Span<float> X, int iniX, int incX)
    {
        unsafe
        {
            fixed (float* ap = &Ap[0]) fixed (float* xp = &X[iniX])
                Native.cblas_stpsv(Layout, UPLO, TransA, Diag, N, ap, xp, incX);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tpsv(ref Layout Layout, ref UpLo UPLO,
        ref Trans TransA, ref Diag Diag,
        ref int N, ref Span<float> Ap, ref Span<float> X, 
        ref int iniX, ref int incX)
    {
        unsafe
        {
            fixed (float* ap = &Ap[0]) fixed (float* xp = &X[iniX])
                Native.cblas_stpsv(Layout, UPLO, TransA, Diag, N, ap, xp, incX);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gemv(Layout Layout,
        Trans TransA, int M, int N,
        double alpha, double[] A, int lda,
        double[] X, int iniX, int incX, double beta,
        double[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_dgemv(Layout, TransA, M, N, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gemv(Layout Layout,
        Trans TransA, int M, int N,
        double alpha, Span<double> A, int lda,
        Span<double> X, int iniX, int incX, double beta,
        Span<double> Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_dgemv(Layout, TransA, M, N, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gemv(ref Layout Layout,
        ref Trans TransA, ref int M, ref int N,
        ref double alpha, ref Span<double> A, ref int lda,
        ref Span<double> X, ref int iniX, ref int incX, ref double beta,
        ref Span<double> Y, ref int iniY, ref int incY)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_dgemv(Layout, TransA, M, N, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gbmv(Layout Layout,
        Trans TransA, int M, int N,
        int KL, int KU, double alpha,
        double[] A, int lda, double[] X, int iniX,
        int incX, double beta, double[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_dgbmv(Layout, TransA, M, N, KL, KU, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gbmv(Layout Layout,
        Trans TransA, int M, int N,
        int KL, int KU, double alpha,
        Span<double> A, int lda, Span<double> X, int iniX,
        int incX, double beta, Span<double> Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_dgbmv(Layout, TransA, M, N, KL, KU, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gbmv(ref Layout Layout,
        ref Trans TransA, ref int M, ref int N,
        ref int KL, ref int KU, ref double alpha,
        ref Span<double> A, ref int lda, ref Span<double> X, ref int iniX,
        ref int incX, ref double beta, ref Span<double> Y, ref int iniY, ref int incY)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_dgbmv(Layout, TransA, M, N, KL, KU, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void trmv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, double[] A, int lda,
        double[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX])
                Native.cblas_dtrmv(Layout, UPLO, TransA, Diag, N, ap, lda, xp, incX);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void trmv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, Span<double> A, int lda,
        Span<double> X, int iniX, int incX)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX])
                Native.cblas_dtrmv(Layout, UPLO, TransA, Diag, N, ap, lda, xp, incX);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void trmv(ref Layout Layout, ref UpLo UPLO,
        ref Trans TransA, ref Diag Diag,
        ref int N, ref Span<double> A, ref int lda,
        ref Span<double> X, ref int iniX, ref int incX)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX])
                Native.cblas_dtrmv(Layout, UPLO, TransA, Diag, N, ap, lda, xp, incX);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tbmv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, int K, double[] A, int lda,
        double[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX])
                Native.cblas_dtbmv(Layout, UPLO, TransA, Diag, N, K, ap, lda, xp, incX);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tbmv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, int K, Span<double> A, int lda,
        Span<double> X, int iniX, int incX)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX])
                Native.cblas_dtbmv(Layout, UPLO, TransA, Diag, N, K, ap, lda, xp, incX);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tbmv(ref Layout Layout, ref UpLo UPLO,
        ref Trans TransA, ref Diag Diag,
        ref int N, ref int K, ref Span<double> A, ref int lda,
        ref Span<double> X, ref int iniX, ref int incX)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX])
                Native.cblas_dtbmv(Layout, UPLO, TransA, Diag, N, K, ap, lda, xp, incX);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tpmv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, double[] Ap, double[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (double* ap = &Ap[0]) fixed (double* xp = &X[iniX])
                Native.cblas_dtpmv(Layout, UPLO, TransA, Diag, N, ap, xp, incX);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tpmv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, Span<double> Ap, Span<double> X, int iniX, int incX)
    {
        unsafe
        {
            fixed (double* ap = &Ap[0]) fixed (double* xp = &X[iniX])
                Native.cblas_dtpmv(Layout, UPLO, TransA, Diag, N, ap, xp, incX);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tpmv(ref Layout Layout, ref UpLo UPLO,
        ref Trans TransA, ref Diag Diag,
        ref int N, ref Span<double> Ap, ref Span<double> X, ref int iniX, ref int incX)
    {
        unsafe
        {
            fixed (double* ap = &Ap[0]) fixed (double* xp = &X[iniX])
                Native.cblas_dtpmv(Layout, UPLO, TransA, Diag, N, ap, xp, incX);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void trsv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, double[] A, int lda, double[] X,
        int iniX, int incX)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX])
                Native.cblas_dtrsv(Layout, UPLO, TransA, Diag, N, ap, lda, xp, incX);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void trsv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, Span<double> A, int lda, Span<double> X,
        int iniX, int incX)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX])
                Native.cblas_dtrsv(Layout, UPLO, TransA, Diag, N, ap, lda, xp, incX);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void trsv(ref Layout Layout, ref UpLo UPLO,
        ref Trans TransA, ref Diag Diag,
        ref int N, ref Span<double> A, ref int lda, ref Span<double> X,
        ref int iniX, ref int incX)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX])
                Native.cblas_dtrsv(Layout, UPLO, TransA, Diag, N, ap, lda, xp, incX);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tbsv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, int K, double[] A, int lda,
        double[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX])
                Native.cblas_dtbsv(Layout, UPLO, TransA, Diag, N, K, ap, lda, xp, incX);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tbsv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, int K, Span<double> A, int lda,
        Span<double> X, int iniX, int incX)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX])
                Native.cblas_dtbsv(Layout, UPLO, TransA, Diag, N, K, ap, lda, xp, incX);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tbsv(ref Layout Layout, ref UpLo UPLO,
        ref Trans TransA, ref Diag Diag,
        ref int N, ref int K, ref Span<double> A, ref int lda,
        ref Span<double> X, ref int iniX, ref int incX)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX])
                Native.cblas_dtbsv(Layout, UPLO, TransA, Diag, N, K, ap, lda, xp, incX);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tpsv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, double[] Ap, double[] X, int iniX, int incX)
    {
        unsafe
        {
            fixed (double* ap = &Ap[0]) fixed (double* xp = &X[iniX])
                Native.cblas_dtpsv(Layout, UPLO, TransA, Diag, N, ap, xp, incX);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tpsv(Layout Layout, UpLo UPLO,
        Trans TransA, Diag Diag,
        int N, Span<double> Ap, Span<double> X, int iniX, int incX)
    {
        unsafe
        {
            fixed (double* ap = &Ap[0]) fixed (double* xp = &X[iniX])
                Native.cblas_dtpsv(Layout, UPLO, TransA, Diag, N, ap, xp, incX);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void tpsv(ref Layout Layout, ref UpLo UPLO,
        ref Trans TransA, ref Diag Diag,
        ref int N, ref Span<double> Ap, ref Span<double> X, ref int iniX, ref int incX)
    {
        unsafe
        {
            fixed (double* ap = &Ap[0]) fixed (double* xp = &X[iniX])
                Native.cblas_dtpsv(Layout, UPLO, TransA, Diag, N, ap, xp, incX);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void symv(Layout Layout, UpLo UPLO,
        int N, float alpha, float[] A,
        int lda, float[] X, int iniX, int incX,
        float beta, float[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_ssymv(Layout, UPLO, N, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void symv(Layout Layout, UpLo UPLO,
        int N, float alpha, Span<float> A,
        int lda, Span<float> X, int iniX, int incX,
        float beta, Span<float> Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_ssymv(Layout, UPLO, N, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void symv(ref Layout Layout, ref UpLo UPLO,
        ref int N, ref float alpha, ref Span<float> A,
        ref int lda, ref Span<float> X, ref int iniX, ref int incX,
        ref float beta, ref Span<float> Y, ref int iniY, ref int incY)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_ssymv(Layout, UPLO, N, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void sbmv(Layout Layout, UpLo UPLO,
        int N, int K, float alpha, float[] A,
        int lda, float[] X, int iniX, int incX,
        float beta, float[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_ssbmv(Layout, UPLO, N, K, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void sbmv(Layout Layout, UpLo UPLO,
        int N, int K, float alpha, Span<float> A,
        int lda, Span<float> X, int iniX, int incX,
        float beta, Span<float> Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_ssbmv(Layout, UPLO, N, K, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void sbmv(ref Layout Layout, ref UpLo UPLO,
        ref int N, ref int K, ref float alpha, ref Span<float> A,
        ref int lda, ref Span<float> X, ref int iniX, ref int incX,
        ref float beta, ref Span<float> Y, ref int iniY, ref int incY)
    {
        unsafe
        {
            fixed (float* ap = &A[0]) fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_ssbmv(Layout, UPLO, N, K, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void spmv(Layout Layout, UpLo UPLO,
        int N, float alpha, float[] Ap,
        float[] X, int iniX, int incX,
        float beta, float[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (float* ap = &Ap[0]) fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_sspmv(Layout, UPLO, N, alpha, ap, xp, incX, beta, yp, incY);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void spmv(Layout Layout, UpLo UPLO,
        int N, float alpha, Span<float> Ap,
        Span<float> X, int iniX, int incX,
        float beta, Span<float> Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (float* ap = &Ap[0]) fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_sspmv(Layout, UPLO, N, alpha, ap, xp, incX, beta, yp, incY);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void spmv(ref Layout Layout, ref UpLo UPLO,
        ref int N, ref float alpha, ref Span<float> Ap,
        ref Span<float> X, ref int iniX, ref int incX,
        ref float beta, ref Span<float> Y, ref int iniY, ref int incY)
    {
        unsafe
        {
            fixed (float* ap = &Ap[0]) fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY])
                Native.cblas_sspmv(Layout, UPLO, N, alpha, ap, xp, incX, beta, yp, incY);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ger(Layout Layout, int M, int N,
        float alpha, float[] X, int iniX, int incX,
        float[] Y, int iniY, int incY, float[] A, int lda)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY]) fixed (float* ap = &A[0])
                Native.cblas_sger(Layout, M, N, alpha, xp, incX, yp, incY, ap, lda);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ger(Layout Layout, int M, int N,
        float alpha, Span<float> X, int iniX, int incX,
        Span<float> Y, int iniY, int incY, Span<float> A, int lda)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY]) fixed (float* ap = &A[0])
                Native.cblas_sger(Layout, M, N, alpha, xp, incX, yp, incY, ap, lda);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ger(ref Layout Layout, ref int M, ref int N,
        ref float alpha, ref Span<float> X, ref int iniX, ref int incX,
        ref Span<float> Y, ref int iniY, ref int incY, ref Span<float> A, ref int lda)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY]) fixed (float* ap = &A[0])
                Native.cblas_sger(Layout, M, N, alpha, xp, incX, yp, incY, ap, lda);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void syr(Layout Layout, UpLo UPLO,
        int N, float alpha, float[] X, int iniX,
        int incX, float[] A, int lda)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* ap = &A[0])
                Native.cblas_ssyr(Layout, UPLO, N, alpha, xp, incX, ap, lda);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void syr(Layout Layout, UpLo UPLO,
        int N, float alpha, Span<float> X, int iniX,
        int incX, Span<float> A, int lda)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* ap = &A[0])
                Native.cblas_ssyr(Layout, UPLO, N, alpha, xp, incX, ap, lda);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void syr(ref Layout Layout, ref UpLo UPLO,
        ref int N, ref float alpha, ref Span<float> X, ref int iniX,
        ref int incX, ref Span<float> A, ref int lda)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* ap = &A[0])
                Native.cblas_ssyr(Layout, UPLO, N, alpha, xp, incX, ap, lda);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void spr(Layout Layout, UpLo UPLO,
        int N, float alpha, float[] X, int iniX,
        int incX, float[] Ap)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* ap = &Ap[0])
                Native.cblas_sspr(Layout, UPLO, N, alpha, xp, incX, ap);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void spr(Layout Layout, UpLo UPLO,
        int N, float alpha, Span<float> X, int iniX,
        int incX, Span<float> Ap)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* ap = &Ap[0])
                Native.cblas_sspr(Layout, UPLO, N, alpha, xp, incX, ap);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void spr(ref Layout Layout, ref UpLo UPLO,
        ref int N, ref float alpha, ref Span<float> X, ref int iniX,
        ref int incX, ref Span<float> Ap)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* ap = &Ap[0])
                Native.cblas_sspr(Layout, UPLO, N, alpha, xp, incX, ap);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void syr2(Layout Layout, UpLo UPLO,
        int N, float alpha, float[] X, int iniX,
        int incX, float[] Y, int iniY, int incY, float[] A,
        int lda)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY]) fixed (float* ap = &A[0])
                Native.cblas_ssyr2(Layout, UPLO, N, alpha, xp, incX, yp, incY, ap, lda);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void syr2(Layout Layout, UpLo UPLO,
        int N, float alpha, Span<float> X, int iniX,
        int incX, Span<float> Y, int iniY, int incY, Span<float> A,
        int lda)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY]) fixed (float* ap = &A[0])
                Native.cblas_ssyr2(Layout, UPLO, N, alpha, xp, incX, yp, incY, ap, lda);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void syr2(ref Layout Layout, ref UpLo UPLO,
        ref int N, ref float alpha, ref Span<float> X, ref int iniX,
        ref int incX, ref Span<float> Y, ref int iniY, ref int incY, ref Span<float> A,
        ref int lda)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY]) fixed (float* ap = &A[0])
                Native.cblas_ssyr2(Layout, UPLO, N, alpha, xp, incX, yp, incY, ap, lda);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void spr2(Layout Layout, UpLo UPLO,
        int N, float alpha, float[] X, int iniX,
        int incX, float[] Y, int iniY, int incY, float[] A)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY]) fixed (float* ap = &A[0])
                Native.cblas_sspr2(Layout, UPLO, N, alpha, xp, incX, yp, incY, ap);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void spr2(Layout Layout, UpLo UPLO,
        int N, float alpha, Span<float> X, int iniX,
        int incX, Span<float> Y, int iniY, int incY, Span<float> A)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY]) fixed (float* ap = &A[0])
                Native.cblas_sspr2(Layout, UPLO, N, alpha, xp, incX, yp, incY, ap);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void spr2(ref Layout Layout, ref UpLo UPLO,
        ref int N, ref float alpha, ref Span<float> X, ref int iniX,
        ref int incX, ref Span<float> Y, ref int iniY, ref int incY, ref Span<float> A)
    {
        unsafe
        {
            fixed (float* xp = &X[iniX]) fixed (float* yp = &Y[iniY]) fixed (float* ap = &A[0])
                Native.cblas_sspr2(Layout, UPLO, N, alpha, xp, incX, yp, incY, ap);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void symv(Layout Layout, UpLo UPLO,
        int N, double alpha, double[] A,
        int lda, double[] X, int iniX, int incX,
        double beta, double[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_dsymv(Layout, UPLO, N, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void symv(Layout Layout, UpLo UPLO,
        int N, double alpha, Span<double> A,
        int lda, Span<double> X, int iniX, int incX,
        double beta, Span<double> Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_dsymv(Layout, UPLO, N, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void symv(ref Layout Layout, ref UpLo UPLO,
        ref int N, ref double alpha, ref Span<double> A,
        ref int lda, ref Span<double> X, ref int iniX, ref int incX,
        ref double beta, ref Span<double> Y, ref int iniY, ref int incY)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_dsymv(Layout, UPLO, N, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void sbmv(Layout Layout, UpLo UPLO,
        int N, int K, double alpha, double[] A,
        int lda, double[] X, int iniX, int incX,
        double beta, double[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_dsbmv(Layout, UPLO, N, K, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void sbmv(Layout Layout, UpLo UPLO,
        int N, int K, double alpha, Span<double> A,
        int lda, Span<double> X, int iniX, int incX,
        double beta, Span<double> Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_dsbmv(Layout, UPLO, N, K, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void sbmv(ref Layout Layout, ref UpLo UPLO,
        ref int N, ref int K, ref double alpha, ref Span<double> A,
        ref int lda, ref Span<double> X, ref int iniX, ref int incX,
        ref double beta, ref Span<double> Y, ref int iniY, ref int incY)
    {
        unsafe
        {
            fixed (double* ap = &A[0]) fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_dsbmv(Layout, UPLO, N, K, alpha, ap, lda, xp, incX, beta, yp, incY);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void spmv(Layout Layout, UpLo UPLO,
        int N, double alpha, double[] Ap,
        double[] X, int iniX, int incX,
        double beta, double[] Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (double* ap = &Ap[0]) fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_dspmv(Layout, UPLO, N, alpha, ap, xp, incX, beta, yp, incY);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void spmv(Layout Layout, UpLo UPLO,
        int N, double alpha, Span<double> Ap,
        Span<double> X, int iniX, int incX,
        double beta, Span<double> Y, int iniY, int incY)
    {
        unsafe
        {
            fixed (double* ap = &Ap[0]) fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_dspmv(Layout, UPLO, N, alpha, ap, xp, incX, beta, yp, incY);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void spmv(ref Layout Layout, ref UpLo UPLO,
        ref int N, ref double alpha, ref Span<double> Ap,
        ref Span<double> X, ref int iniX, ref int incX,
        ref double beta, ref Span<double> Y, ref int iniY, ref int incY)
    {
        unsafe
        {
            fixed (double* ap = &Ap[0]) fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY])
                Native.cblas_dspmv(Layout, UPLO, N, alpha, ap, xp, incX, beta, yp, incY);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ger(Layout Layout, int M, int N,
        double alpha, double[] X, int iniX, int incX,
        double[] Y, int iniY, int incY, double[] A, int lda)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY]) fixed (double* ap = &A[0])
                Native.cblas_dger(Layout, M, N, alpha, xp, incX, yp, incY, ap, lda);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ger(Layout Layout, int M, int N,
        double alpha, Span<double> X, int iniX, int incX,
        Span<double> Y, int iniY, int incY, Span<double> A, int lda)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY]) fixed (double* ap = &A[0])
                Native.cblas_dger(Layout, M, N, alpha, xp, incX, yp, incY, ap, lda);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ger(ref Layout Layout, ref int M, ref int N,
        ref double alpha, ref Span<double> X, ref int iniX, ref int incX,
        ref Span<double> Y, ref int iniY, ref int incY, ref Span<double> A, ref int lda)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY]) fixed (double* ap = &A[0])
                Native.cblas_dger(Layout, M, N, alpha, xp, incX, yp, incY, ap, lda);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void syr(Layout Layout, UpLo UPLO,
        int N, double alpha, double[] X, int iniX,
        int incX, double[] A, int lda)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* ap = &A[0])
                Native.cblas_dsyr(Layout, UPLO, N, alpha, xp, incX, ap, lda);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void syr(Layout Layout, UpLo UPLO,
        int N, double alpha, Span<double> X, int iniX,
        int incX, Span<double> A, int lda)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* ap = &A[0])
                Native.cblas_dsyr(Layout, UPLO, N, alpha, xp, incX, ap, lda);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void syr(ref Layout Layout, ref UpLo UPLO,
        ref int N, ref double alpha, ref Span<double> X, ref int iniX,
        ref int incX, ref Span<double> A, ref int lda)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* ap = &A[0])
                Native.cblas_dsyr(Layout, UPLO, N, alpha, xp, incX, ap, lda);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void spr(Layout Layout, UpLo UPLO,
        int N, double alpha, double[] X, int iniX,
        int incX, double[] Ap)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* ap = &Ap[0])
                Native.cblas_dspr(Layout, UPLO, N, alpha, xp, incX, ap);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void spr(Layout Layout, UpLo UPLO,
        int N, double alpha, Span<double> X, int iniX,
        int incX, Span<double> Ap)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* ap = &Ap[0])
                Native.cblas_dspr(Layout, UPLO, N, alpha, xp, incX, ap);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void spr(ref Layout Layout, ref UpLo UPLO,
        ref int N, ref double alpha, ref Span<double> X, ref int iniX,
        ref int incX, ref Span<double> Ap)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* ap = &Ap[0])
                Native.cblas_dspr(Layout, UPLO, N, alpha, xp, incX, ap);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void syr2(Layout Layout, UpLo UPLO,
        int N, double alpha, double[] X, int iniX,
        int incX, double[] Y, int iniY, int incY, double[] A,
        int lda)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY]) fixed (double* ap = &A[0])
                Native.cblas_dsyr2(Layout, UPLO, N, alpha, xp, incX, yp, incY, ap, lda);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void syr2(Layout Layout, UpLo UPLO,
        int N, double alpha, Span<double> X, int iniX,
        int incX, Span<double> Y, int iniY, int incY, Span<double> A,
        int lda)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY]) fixed (double* ap = &A[0])
                Native.cblas_dsyr2(Layout, UPLO, N, alpha, xp, incX, yp, incY, ap, lda);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void syr2(ref Layout Layout, ref UpLo UPLO,
        ref int N, ref double alpha, ref Span<double> X, ref int iniX,
        ref int incX, ref Span<double> Y, ref int iniY, ref int incY, ref Span<double> A,
        ref int lda)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY]) fixed (double* ap = &A[0])
                Native.cblas_dsyr2(Layout, UPLO, N, alpha, xp, incX, yp, incY, ap, lda);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void spr2(Layout Layout, UpLo UPLO,
        int N, double alpha, double[] X, int iniX,
        int incX, double[] Y, int iniY, int incY, double[] A)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY]) fixed (double* ap = &A[0])
                Native.cblas_dspr2(Layout, UPLO, N, alpha, xp, incX, yp, incY, ap);
        }
    }

#if NET6_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void spr2(Layout Layout, UpLo UPLO,
        int N, double alpha, Span<double> X, int iniX,
        int incX, Span<double> Y, int iniY, int incY, Span<double> A)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY]) fixed (double* ap = &A[0])
                Native.cblas_dspr2(Layout, UPLO, N, alpha, xp, incX, yp, incY, ap);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void spr2(ref Layout Layout, ref UpLo UPLO,
        ref int N, ref double alpha, ref Span<double> X, ref int iniX,
        ref int incX, ref Span<double> Y, ref int iniY, ref int incY, ref Span<double> A)
    {
        unsafe
        {
            fixed (double* xp = &X[iniX]) fixed (double* yp = &Y[iniY]) fixed (double* ap = &A[0])
                Native.cblas_dspr2(Layout, UPLO, N, alpha, xp, incX, yp, incY, ap);
        }
    }
#endif

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gemm(Layout Layout, Trans TransA,
        Trans TransB, int M, int N,
        int K, float alpha, float[] A,
        int lda, float[] B, int ldb,
        float beta, float[] C, int ldc)
        => Native.cblas_sgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gemmt(Layout Layout, UpLo UPLO,
        Trans TransA, Trans TransB,
        int N, int K,
        float alpha, float[] A, int lda,
        float[] B, int ldb, float beta,
        float[] C, int ldc)
        => Native.cblas_sgemmt(Layout, UPLO, TransA, TransB, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void symm(Layout Layout, Side Side,
        UpLo UPLO, int M, int N,
        float alpha, float[] A, int lda,
        float[] B, int ldb, float beta,
        float[] C, int ldc)
        => Native.cblas_ssymm(Layout, Side, UPLO, M, N, alpha, A, lda, B, ldb, beta, C, ldc);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void syrk(Layout Layout, UpLo UPLO,
        Trans Trans, int N, int K,
        float alpha, float[] A, int lda,
        float beta, float[] C, int ldc)
        => Native.cblas_ssyrk(Layout, UPLO, Trans, N, K, alpha, A, lda, beta, C, ldc);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void syr2k(Layout Layout, UpLo UPLO,
        Trans Trans, int N, int K,
        float alpha, float[] A, int lda,
        float[] B, int ldb, float beta,
        float[] C, int ldc)
        => Native.cblas_ssyr2k(Layout, UPLO, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void trmm(Layout Layout, Side Side,
        UpLo UPLO, Trans TransA,
        Diag Diag, int M, int N,
        float alpha, float[] A, int lda,
        float[] B, int ldb)
        => Native.cblas_strmm(Layout, Side, UPLO, TransA, Diag, M, N, alpha, A, lda, B, ldb);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void trsm(Layout Layout, Side Side,
        UpLo UPLO, Trans TransA,
        Diag Diag, int M, int N,
        float alpha, float[] A, int lda,
        float[] B, int ldb)
        => Native.cblas_strsm(Layout, Side, UPLO, TransA, Diag, M, N, alpha, A, lda, B, ldb);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gemm(Layout Layout, Trans TransA,
        Trans TransB, int M, int N,
        int K, double alpha, double[] A,
        int lda, double[] B, int ldb,
        double beta, double[] C, int ldc)
        => Native.cblas_dgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void gemmt(Layout Layout, UpLo UPLO,
        Trans TransA, Trans TransB,
        int N, int K,
        double alpha, double[] A, int lda,
        double[] B, int ldb, double beta,
        double[] C, int ldc)
        => Native.cblas_dgemmt(Layout, UPLO, TransA, TransB, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void symm(Layout Layout, Side Side,
        UpLo UPLO, int M, int N,
        double alpha, double[] A, int lda,
        double[] B, int ldb, double beta,
        double[] C, int ldc)
        => Native.cblas_dsymm(Layout, Side, UPLO, M, N, alpha, A, lda, B, ldb, beta, C, ldc);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void syrk(Layout Layout, UpLo UPLO,
        Trans Trans, int N, int K,
        double alpha, double[] A, int lda,
        double beta, double[] C, int ldc)
        => Native.cblas_dsyrk(Layout, UPLO, Trans, N, K, alpha, A, lda, beta, C, ldc);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void syr2k(Layout Layout, UpLo UPLO,
        Trans Trans, int N, int K,
        double alpha, double[] A, int lda,
        double[] B, int ldb, double beta,
        double[] C, int ldc)
        => Native.cblas_dsyr2k(Layout, UPLO, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void trmm(Layout Layout, Side Side,
        UpLo UPLO, Trans TransA,
        Diag Diag, int M, int N,
        double alpha, double[] A, int lda,
        double[] B, int ldb)
        => Native.cblas_dtrmm(Layout, Side, UPLO, TransA, Diag, M, N, alpha, A, lda, B, ldb);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void trsm(Layout Layout, Side Side,
        UpLo UPLO, Trans TransA,
        Diag Diag, int M, int N,
        double alpha, double[] A, int lda,
        double[] B, int ldb)
        => Native.cblas_dtrsm(Layout, Side, UPLO, TransA, Diag, M, N, alpha, A, lda, B, ldb);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void imatcopy(LayoutChar ordering, TransChar trans, int rows, int cols, double alpha, double[] A, int lda, int ldb)
        => Native.MKL_Dimatcopy(ordering, trans, rows, cols, alpha, A, lda, ldb);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void imatcopy(LayoutChar ordering, TransChar trans, int rows, int cols, float alpha, float[] A, int lda, int ldb)
        => Native.MKL_Simatcopy(ordering, trans, rows, cols, alpha, A, lda, ldb);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void omatcopy(LayoutChar ordering, TransChar trans, int rows, int cols, double alpha, double[] A, int lda, double[] B, int ldb)
        => Native.MKL_Domatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void omatcopy(LayoutChar ordering, TransChar trans, int rows, int cols, float alpha, float[] A, int lda, float[] B, int ldb)
        => Native.MKL_Somatcopy(ordering, trans, rows, cols, alpha, A, lda, B, ldb);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void omatadd(LayoutChar ordering, TransChar transa, TransChar transb, int rows, int cols, double alpha, double[] A,
        int lda, double beta, double[] B, int ldb, double[] C, int ldc)
        => Native.MKL_Domatadd(ordering, transa, transb, rows, cols, alpha, A, lda, beta, B, ldb, C, ldc);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void omatadd(LayoutChar ordering, TransChar transa, TransChar transb, int rows, int cols, float alpha, float[] A,
        int lda, float beta, float[] B, int ldb, float[] C, int ldc)
        => Native.MKL_Somatadd(ordering, transa, transb, rows, cols, alpha, A, lda, beta, B, ldb, C, ldc);


    public static unsafe class Native
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern float cblas_sdot(int N, float* X, int incX, float* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern float cblas_sdoti(int N, float[] X, int[] indx, float[] Y);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern double cblas_ddot(int N, double* X, int incX, double* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern double cblas_ddoti(int N, double[] X, int[] indx, double[] Y);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern double cblas_dsdot(int N, float* X, int incX, float* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern float cblas_sdsdot(int N, float sb, float* X, int incX, float* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern float cblas_snrm2(int N, float* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern float cblas_sasum(int N, float* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern double cblas_dnrm2(int N, double* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern double cblas_dasum(int N, double* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cblas_isamax(int N, float* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cblas_idamax(int N, double* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cblas_isamin(int N, float* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cblas_idamin(int N, double* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_sswap(int N, float* X, int incX, float* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_scopy(int N, float* X, int incX, float* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_saxpy(int N, float a, float* X, int incX, float* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_saxpby(int N, float alpha, float* X, int incX, float beta, float* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_saxpyi(int N, float alpha, float[] X, int[] indx, float[] Y);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_sgthr(int N, float[] Y, float[] X, int[] indx);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_sgthrz(int N, float[] Y, float[] X, int[] indx);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_ssctr(int N, float[] X, int[] indx, float[] Y);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_srotg(ref float a, ref float b, ref float c, ref float s);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dswap(int N, double* X, int incX, double* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dcopy(int N, double* X, int incX, double* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_daxpy(int N, double a, double* X, int incX, double* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_daxpby(int N, double alpha, double* X, int incX, double beta, double* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_daxpyi(int N, double alpha, double[] X, int[] indx, double[] Y);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dgthr(int N, double[] Y, double[] X, int[] indx);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dgthrz(int N, double[] Y, double[] X, int[] indx);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dsctr(int N, double[] X, int[] indx, double[] Y);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_drotg(ref double a, ref double b, ref double c, ref double s);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_srotmg(ref float d1, ref float d2, ref float x1, float y1, float* param);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_srot(int N, float* X, int incX, float* Y, int incY, float c, float s);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_sroti(int N, float[] X, int[] indx, float[] Y, float c, float s);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_srotm(int N, float* X, int incX, float* Y, int incY, float* param);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_drotmg(ref double d1, ref double d2, ref double x1, double y1, double* param);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_drot(int N, double* X, int incX, double* Y, int incY, double c, double s);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_drotm(int N, double* X, int incX, double* Y, int incY, double* param);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_droti(int N, double[] X, int[] indx, double[] Y, double c, double s);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_sscal(int N, float a, float* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dscal(int N, double a, double* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_sgemv(Layout Layout,
            Trans TransA, int M, int N,
            float alpha, float* A, int lda,
            float* X, int incX, float beta,
            float* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_sgbmv(Layout Layout,
            Trans TransA, int M, int N,
            int KL, int KU, float alpha,
            float* A, int lda, float* X,
            int incX, float beta, float* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_strmv(Layout Layout, UpLo UPLO,
            Trans TransA, Diag Diag,
            int N, float* A, int lda,
            float* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_stbmv(Layout Layout, UpLo UPLO,
            Trans TransA, Diag Diag,
            int N, int K, float* A, int lda,
            float* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_stpmv(Layout Layout, UpLo UPLO,
            Trans TransA, Diag Diag,
            int N, float* Ap, float* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_strsv(Layout Layout, UpLo UPLO,
            Trans TransA, Diag Diag,
            int N, float* A, int lda, float* X,
            int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_stbsv(Layout Layout, UpLo UPLO,
            Trans TransA, Diag Diag,
            int N, int K, float* A, int lda,
            float* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_stpsv(Layout Layout, UpLo UPLO,
            Trans TransA, Diag Diag,
            int N, float* Ap, float* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dgemv(Layout Layout,
            Trans TransA, int M, int N,
            double alpha, double* A, int lda,
            double* X, int incX, double beta,
            double* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dgbmv(Layout Layout,
            Trans TransA, int M, int N,
            int KL, int KU, double alpha,
            double* A, int lda, double* X,
            int incX, double beta, double* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dtrmv(Layout Layout, UpLo UPLO,
            Trans TransA, Diag Diag,
            int N, double* A, int lda,
            double* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dtbmv(Layout Layout, UpLo UPLO,
            Trans TransA, Diag Diag,
            int N, int K, double* A, int lda,
            double* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dtpmv(Layout Layout, UpLo UPLO,
            Trans TransA, Diag Diag,
            int N, double* Ap, double* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dtrsv(Layout Layout, UpLo UPLO,
            Trans TransA, Diag Diag,
            int N, double* A, int lda, double* X,
            int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dtbsv(Layout Layout, UpLo UPLO,
            Trans TransA, Diag Diag,
            int N, int K, double* A, int lda,
            double* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dtpsv(Layout Layout, UpLo UPLO,
            Trans TransA, Diag Diag,
            int N, double* Ap, double* X, int incX);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_ssymv(Layout Layout, UpLo UPLO,
            int N, float alpha, float* A,
            int lda, float* X, int incX,
            float beta, float* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_ssbmv(Layout Layout, UpLo UPLO,
            int N, int K, float alpha, float* A,
            int lda, float* X, int incX,
            float beta, float* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_sspmv(Layout Layout, UpLo UPLO,
            int N, float alpha, float* Ap,
            float* X, int incX,
            float beta, float* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_sger(Layout Layout, int M, int N,
            float alpha, float* X, int incX,
            float* Y, int incY, float* A, int lda);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_ssyr(Layout Layout, UpLo UPLO,
            int N, float alpha, float* X,
            int incX, float* A, int lda);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_sspr(Layout Layout, UpLo UPLO,
            int N, float alpha, float* X,
            int incX, float* Ap);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_ssyr2(Layout Layout, UpLo UPLO,
            int N, float alpha, float* X,
            int incX, float* Y, int incY, float* A,
            int lda);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_sspr2(Layout Layout, UpLo UPLO,
            int N, float alpha, float* X,
            int incX, float* Y, int incY, float* A);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dsymv(Layout Layout, UpLo UPLO,
            int N, double alpha, double* A,
            int lda, double* X, int incX,
            double beta, double* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dsbmv(Layout Layout, UpLo UPLO,
            int N, int K, double alpha, double* A,
            int lda, double* X, int incX,
            double beta, double* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dspmv(Layout Layout, UpLo UPLO,
            int N, double alpha, double* Ap,
            double* X, int incX,
            double beta, double* Y, int incY);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dger(Layout Layout, int M, int N,
            double alpha, double* X, int incX,
            double* Y, int incY, double* A, int lda);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dsyr(Layout Layout, UpLo UPLO,
                int N, double alpha, double* X,
                int incX, double* A, int lda);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dspr(Layout Layout, UpLo UPLO,
            int N, double alpha, double* X,
            int incX, double* Ap);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dsyr2(Layout Layout, UpLo UPLO,
            int N, double alpha, double* X,
            int incX, double* Y, int incY, double* A,
            int lda);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dspr2(Layout Layout, UpLo UPLO,
            int N, double alpha, double* X,
            int incX, double* Y, int incY, double* A);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_sgemm(Layout Layout, Trans TransA,
            Trans TransB, int M, int N,
            int K, float alpha, float[] A,
            int lda, float[] B, int ldb,
            float beta, float[] C, int ldc);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_sgemmt(Layout Layout, UpLo UPLO,
            Trans TransA, Trans TransB,
            int N, int K,
            float alpha, float[] A, int lda,
            float[] B, int ldb, float beta,
            float[] C, int ldc);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_ssymm(Layout Layout, Side Side,
            UpLo UPLO, int M, int N,
            float alpha, float[] A, int lda,
            float[] B, int ldb, float beta,
            float[] C, int ldc);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_ssyrk(Layout Layout, UpLo UPLO,
            Trans Trans, int N, int K,
            float alpha, float[] A, int lda,
            float beta, float[] C, int ldc);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_ssyr2k(Layout Layout, UpLo UPLO,
            Trans Trans, int N, int K,
            float alpha, float[] A, int lda,
            float[] B, int ldb, float beta,
            float[] C, int ldc);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_strmm(Layout Layout, Side Side,
            UpLo UPLO, Trans TransA,
            Diag Diag, int M, int N,
            float alpha, float[] A, int lda,
            float[] B, int ldb);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_strsm(Layout Layout, Side Side,
            UpLo UPLO, Trans TransA,
            Diag Diag, int M, int N,
            float alpha, float[] A, int lda,
            float[] B, int ldb);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dgemm(Layout Layout, Trans TransA,
            Trans TransB, int M, int N,
            int K, double alpha, double[] A,
            int lda, double[] B, int ldb,
            double beta, double[] C, int ldc);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dgemmt(Layout Layout, UpLo UPLO,
            Trans TransA, Trans TransB,
            int N, int K,
            double alpha, double[] A, int lda,
            double[] B, int ldb, double beta,
            double[] C, int ldc);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dsymm(Layout Layout, Side Side,
            UpLo UPLO, int M, int N,
            double alpha, double[] A, int lda,
            double[] B, int ldb, double beta,
            double[] C, int ldc);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dsyrk(Layout Layout, UpLo UPLO,
            Trans Trans, int N, int K,
            double alpha, double[] A, int lda,
            double beta, double[] C, int ldc);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dsyr2k(Layout Layout, UpLo UPLO,
            Trans Trans, int N, int K,
            double alpha, double[] A, int lda,
            double[] B, int ldb, double beta,
            double[] C, int ldc);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dtrmm(Layout Layout, Side Side,
            UpLo UPLO, Trans TransA,
            Diag Diag, int M, int N,
            double alpha, double[] A, int lda,
            double[] B, int ldb);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cblas_dtrsm(Layout Layout, Side Side,
            UpLo UPLO, Trans TransA,
            Diag Diag, int M, int N,
            double alpha, double[] A, int lda,
            double[] B, int ldb);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void MKL_Dimatcopy(LayoutChar ordering, TransChar trans, int rows, int cols, double alpha, double[] A, int lda, int ldb);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void MKL_Simatcopy(LayoutChar ordering, TransChar trans, int rows, int cols, float alpha, float[] A, int lda, int ldb);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void MKL_Domatcopy(LayoutChar ordering, TransChar trans, int rows, int cols, double alpha, double[] A, int lda, double[] B, int ldb);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void MKL_Somatcopy(LayoutChar ordering, TransChar trans, int rows, int cols, float alpha, float[] A, int lda, float[] B, int ldb);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void MKL_Domatadd(LayoutChar ordering, TransChar transa, TransChar transb, int rows, int cols, double alpha, double[] A,
            int lda, double beta, double[] B, int ldb, double[] C, int ldc);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [DllImport(MKL.DLL, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void MKL_Somatadd(LayoutChar ordering, TransChar transa, TransChar transb, int rows, int cols, float alpha, float[] A,
            int lda, float beta, float[] B, int ldb, float[] C, int ldc);
    }

}