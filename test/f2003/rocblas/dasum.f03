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

program rocblas_dasum_test

    use iso_c_binding
    use hipfort
    use hipfort_check
    use hipfort_rocblas

    implicit none

    ! N must be a multiple of 4: the input repeats with period 4 and the
    ! expected value below is a closed form in the number of whole periods.
    integer, parameter :: N = 12000
    integer, parameter :: bytes_per_element = 8 ! double precision
    integer(c_size_t), parameter :: Nbytes = N * bytes_per_element

    type(c_ptr) :: dx = c_null_ptr

    real(c_double),allocatable,target,dimension(:) :: hx
    real(c_double),target :: res

    real(c_double) :: res_exact
    real(c_double) :: error
    real(c_double), parameter :: error_max = 10 * epsilon(error_max)

    type(c_ptr) :: rocblas_handle

    integer :: i, m, sgn

    write(*,"(a)",advance="no") "-- Running test 'dasum' (Fortran 2003 interfaces) - "

    ! Create rocblas handle
    call rocblasCheck(rocblas_create_handle(rocblas_handle))
    call rocblasCheck(rocblas_set_pointer_mode(rocblas_handle, 0)) ! host pointer mode

    ! Allocate host-side memory
    allocate(hx(N))

    ! Initialize host memory. Magnitudes cycle 1,2,3,4 so a routine that reads
    ! the wrong elements gets wrong values; a constant vector could not catch
    ! that. Signs alternate so a missing abs() is visible: the signed sum is
    ! -6000, of the opposite sign to the correct +30000.
    do i = 1, N
        m   = 1 + mod(i - 1, 4)
        sgn = merge(1, -1, mod(i, 2) == 1)
        hx(i) = sgn * m
    end do

    ! sum(|x|) = (1+2+3+4) per period of 4. Every partial sum is an integer
    ! well below 2**24, so the device reduction is exact whatever order it
    ! accumulates in - unlike a ramp, whose sum would not be representable.
    res_exact = 10.0d0 * (N / 4)                      ! = 30000

    ! Allocate device-side memory
    call hipCheck(hipMalloc(dx, Nbytes))

    ! Transfer data from host to device memory
    call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

    ! Call rocblas function
    res = 0.0d0
    call rocblasCheck(rocblas_dasum(rocblas_handle, N, dx, 1, c_loc(res)))
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

end program rocblas_dasum_test
