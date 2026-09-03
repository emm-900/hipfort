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

program rocblas_scasum_test

    use iso_c_binding
    use hipfort
    use hipfort_check
    use hipfort_rocblas

    implicit none

    ! N must be a multiple of 4: the input repeats with period 4 and the
    ! expected value below is a closed form in the number of whole periods.
    integer, parameter :: N = 12000

    complex(c_float_complex),allocatable,target,dimension(:) :: hx

    ! Mixed typing: a COMPLEX vector reduced to a REAL result. Note scasum and
    ! sasum have byte-identical C signatures, differing only in which symbol
    ! they bind to, so the expected value below has to be able to tell them
    ! apart.
    real(c_float),target :: res

    complex(c_float_complex),pointer,dimension(:) :: dx

    real(c_float) :: res_exact
    real(c_float) :: error
    real(c_float), parameter :: error_max = 10 * epsilon(error_max)

    type(c_ptr) :: rocblas_handle

    integer :: i, m, sgn

    write(*,"(a)",advance="no") "-- Running test 'scasum' (Fortran 2008 interfaces) - "

    ! Create rocblas handle
    call rocblasCheck(rocblas_create_handle(rocblas_handle))
    call rocblasCheck(rocblas_set_pointer_mode(rocblas_handle, 0)) ! host pointer mode

    ! Allocate host-side memory
    allocate(hx(N))

    ! Initialize host memory. Magnitudes cycle 1,2,3,4 so a routine that reads
    ! the wrong elements gets wrong values. The components are given the ratio
    ! 3:-4 so the two readings of complex asum differ (see res_exact), and
    ! opposite signs so dropping either component's abs() is visible.
    do i = 1, N
        m   = 1 + mod(i - 1, 4)
        sgn = merge(1, -1, mod(i, 2) == 1)
        hx(i) = cmplx(3 * sgn * m, -4 * sgn * m, kind=c_float_complex)
    end do

    ! scasum is sum(|Re| + |Im|), NOT sum(|x|). Per period of 4 that is
    ! (3+4) * (1+2+3+4) = 70. The 3:4 ratio separates the two readings by 40%
    ! (210000 against 150000), so the wrong convention cannot silently pass.
    ! Every partial sum is an integer well below 2**24, so the device
    ! reduction is exact whatever order it accumulates in.
    res_exact = 70.0 * (N / 4)                        ! = 210000

    ! Allocate device-side memory
    ! Transfer data from host to device memory
    call hipCheck(hipMalloc(dx, source=hx))

    ! Call rocblas function. Only x has pointer overloads; the result is still
    ! passed as a c_ptr to a host scalar.
    res = 0.0
    call rocblasCheck(rocblas_scasum(rocblas_handle, N, dx, 1, c_loc(res)))
    call hipCheck(hipDeviceSynchronize()) ! res now valid host-side

    ! Verification
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

end program rocblas_scasum_test
