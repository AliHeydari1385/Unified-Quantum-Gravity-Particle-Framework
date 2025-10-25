!===========================================================
program test

!===========================================================
!ACT 3-year CMB-only likelihood code
!v2p2 updates to include improved beams (and the improved binning used in v2)
!v2 includes option to add SPT Story et al 2012 data, foreground-marginalized
!E. Calabrese, J. Dunkley Jan 2013. Based on WMAP likelihood codes.
!===========================================================

use actlite_3yr_like

implicit none

real(8), dimension(:), allocatable :: cl_tt
character(LEN=128) :: filename
real(8)            :: like
integer            :: lun, il, dummy,i,j

!---------------------------------------------------

print *,""
print *,"ACT likelihood test"
print *,"==================================="
call act_likelihood_init

!---------------------------------------------------
! read in test Cls
!---------------------------------------------------

filename = trim(ACT_data_dir)//'cmb_bftot_lensedCls.dat'
write(*,*)"Reading in Cls from: ",trim(filename)
call get_free_lun( lun )

allocate(cl_tt(2:tt_lmax))
cl_tt(2:tt_lmax)=0.d0

open(unit=lun,file=filename,action='read',status='old')
do il=2,tt_lmax
   read(lun,*)dummy,cl_tt(il)
enddo
close(lun)
!Theory Cls required up to lmax=3750

call act_likelihood_compute(cl_tt,like)

write(*,*) '-------------------------------------'
write(*,*) '         -2lnL = ', 2.*like
write(*,*) ''
write(*,*) 'Expected -2lnL =    77.3844477474373 for 68 bins (for ACT+SPT) '
write(*,*) 'Expected -2lnL =    38.4232643952823 for 42 bins (for ACT only) '
write(*,*) 'Expected -2lnL =    57.3883424852386 for 47 bins (for SPT only)' 
write(*,*) '-------------------------------------'
write(*,*) ''

end program test
	
