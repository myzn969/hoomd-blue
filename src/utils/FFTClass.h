/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __FFT_CLASS_H__
#define __FFT_CLASS_H__

#include "ParticleData.h"

#ifdef ENABLE_FFT

/*! \file FFTClass.h
    \brief Declares the FFTClass class
*/

//! provides a general interface for fft in HOOMD
/*! The three member functions are respectively, the 3D fft of a complex matrix, the fft of
    a real matrix, and the fft of a complex matrix whose fft is a real matrix. The data
    types are three dimensional matrices.
    \note This class is abstract and therefore cannot be instantiated.
*/
class FFTClass
    {
    public:
    
        //! Complex FFT
        /*! \param N_x number of grid points in the x axis
            \param N_y number of grid points in the y axis
            \param N_z number of grid points in the z axis
            \param in 3D complex matrix labelled as a 1D dimensional matrix
            \param out 3D complex matrix labelled as a 1D dimensional matrix
            \param sig specify if the FFT is forwards or backward
        
            3D FFT of complex matrix in, result stored in matrix out, sign=-1 (forward)
            or +1 (backward)
        
            \todo document me
        */
        virtual void cmplx_fft(unsigned int N_x,unsigned int N_y,unsigned int N_z,CScalar *in,CScalar *out,int sig)=0;
        
        //! Real to complex FFT
        /*! \param N_x number of grid points in the x axis
            \param N_y number of grid points in the y axis
            \param N_z number of grid points in the z axis
            \param in 3D scalar matrix labelled as a 1D dimensional matrix
            \param out 3D complex matrix labelled as a 1D dimensional matrix
        
            3D FFT of real matrix in, result stored in matrix out, forward is implictly assumed
        */
        virtual void real_to_compl_fft(unsigned int N_x,unsigned int N_y,unsigned int N_z,Scalar *in,CScalar *out)=0;
        
        //! Complex to real FFT
        /*! \param N_x number of grid points in the x axis
            \param N_y number of grid points in the y axis
            \param N_z number of grid points in the z axis
            \param in 3D complex scalar matrix labelled as a 1D dimensional matrix
            \param out 3D scalar matrix labelled as a 1D dimensional matrix
        
            3D FFT of complex matrix in, result is real and stored in matrix
            out, backward is implictly assumed
        
            \todo document me
        */
        virtual void compl_to_real_fft(unsigned int N_x,unsigned int N_y,unsigned int N_z,CScalar *in,Scalar *out)=0;
        
        //! Virtual destructor
        virtual ~FFTClass() {}
        
    };

#endif
#endif

