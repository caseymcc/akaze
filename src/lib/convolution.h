// Copyright (C) 2014 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_

#include <Eigen/Dense>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowMatrixXf;

enum BorderType {
  REFLECT = 0,
  REPLICATE = 1,
  DEFAULT = REFLECT
};

// Performs separable convolution using two filters of the same size.
void SeparableConvolution2d(const RowMatrixXf& image,
                            const Eigen::RowVectorXf& kernel_x,
                            const Eigen::RowVectorXf& kernel_y,
                            const BorderType& border_type,
                            RowMatrixXf* out);

// Computes the image derivative using the Scharr filter.
void ScharrDerivative(const RowMatrixXf& image,
                      const int x_deg,
                      const int y_deg,
                      const int size,
                      const bool normalize,
                      RowMatrixXf* out);

void GaussianBlur(const RowMatrixXf& image,
                  const double sigma,
                  RowMatrixXf* out);

///
/// Convolves two matrices
///
template <typename Derived1_, typename Derived2_>
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> convolveMatrix(const Eigen::MatrixBase<Derived1_>& matrix1, const Eigen::MatrixBase<Derived2_> &matrix2)
{
    size_t cols=matrix1.cols()+matrix2.cols()-1;
    size_t rows=matrix1.rows()+matrix2.rows()-1;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> output=Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Zero(rows, cols);

    size_t matrix1Row=0;
    size_t matrix2Row=matrix2.rows()-1;
    size_t rowSize=1;

    for(size_t y=0; y<rows; ++y)
    {
        size_t matrix1Col=0;
        size_t matrix2Col=matrix2.cols()-1;
        size_t colSize=1;

        for(size_t x=0; x<cols; ++x)
        {
            Derived1_::Scalar b=(matrix1.block(matrix1Row, matrix1Col, rowSize, colSize).cwiseProduct(matrix2.block(matrix2Row, matrix2Col, rowSize, colSize))).sum();
            output.coeffRef(y, x)=b;

            if((matrix1Col<=matrix1.cols()-1)&&(matrix2Col>=0))
            {
                if((colSize<matrix1.cols())&&(colSize<matrix2.cols()))
                    colSize++;
                matrix2Col--;
            }
            else
            {
                matrix1Col++;
                if(colSize > matrix1.cols()-matrix1Col)
                    colSize--;
            }
        }

        if((matrix1Row<=matrix1.rows()-1)&&(matrix2Row>0))
        {
            if((rowSize<matrix1.rows())&&(rowSize<matrix2.rows()))
                rowSize++;
            matrix2Row--;
        }
        else
        {
            matrix1Row++;
            if(rowSize > matrix1.rows()-matrix1Row)
                rowSize--;
        }
    }

    return output;
}

#endif  // CONVOLUTION_H_
