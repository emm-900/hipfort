!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2020-2026 Advanced Micro Devices, Inc.
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

program rocblas_scnrm2_test

    use iso_c_binding
    use hipfort
    use hipfort_check
    use hipfort_rocblas

    implicit none

    ! N must be a multiple of 4: the input repeats with period 4 and the
    ! expected value below is a closed form in the number of whole periods.
    integer, parameter :: N = 12000

    complex(c_float_complex),allocatable,target,dimension(:) :: hx

    ! Mixed typing: a COMPLEX vector reduced to a REAL result. The result kind
    ! follows the routine's precision letter, not the element type.
    real(c_float),target :: res

    complex(c_float_complex),pointer,dimension(:) :: dx

    real(c_float) :: res_exact
    real(c_float) :: error
    real(c_float), parameter :: error_max = 10 * epsilon(error_max)

    type(c_ptr) :: rocblas_handle

    integer :: i, m, sgn

    write(*,"(a)",advance="no") "-- Running test 'scnrm2' (Fortran 2008 interfaces) - "

    ! Create rocblas handle
    call rocblasCheck(rocblas_create_handle(rocblas_handle))
    call rocblasCheck(rocblas_set_pointer_mode(rocblas_handle, 0)) ! host pointer mode

    ! Allocate host-side memory
    allocate(hx(N))

    ! Initialize host memory. Magnitudes cycle 1,2,3,4 so a routine that reads
    ! the wrong elements gets wrong values. The 3:-4 component ratio makes the
    ! modulus 5m an integer, which is what keeps the sum of squares - and so
    ! the square root below - exact.
    do i = 1, N
        m   = 1 + mod(i - 1, 4)
        sgn = merge(1, -1, mod(i, 2) == 1)
        hx(i) = cmplx(3 * sgn * m, -4 * sgn * m, kind=c_float_complex)
    end do

    ! scnrm2 = sqrt(sum(Re**2 + Im**2)). Per period of 4 that is
    ! (9+16) * (1+4+9+16) = 750, and 750 * (N/4) = 2250000 = 1500**2, so the
    ! square root is exact too. Dropping either component would give 900 or
    ! 1200 rather than 1500.
    res_exact = sqrt(750.0 * (N / 4))                 ! = 1500

    ! Allocate device-side memory
    ! Transfer data from host to device memory
    call hipCheck(hipMalloc(dx, source=hx))

    ! Call rocblas function. Only x has pointer overloads; the result is still
    ! passed as a c_ptr to a host scalar.
    res = 0.0
    call rocblasCheck(rocblas_scnrm2(rocblas_handle, N, dx, 1, c_loc(res)))
    call hipCheck(hipDeviceSynchronize()) ! res now valid host-side

    ! Verification. The tolerance is kept because nrm2 implementations may
    ! rescale internally to avoid overflow, which can cost a few ulp even
    ! though the mathematical answer here is an integer.
    error = abs((res_exact - res) / res_exact)
    if(error .gt. error_max) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " result = ", res
        call exit(1)
    end if

    ! Cleanup
    call hipCheck(hipFree(dx))
    deallocate(hx)
    call rocblasCheck(rocblas_destroy_handle(rocblas_handle))

    write(*,*) "PASSED!"

end program rocblas_scnrm2_test
