! ===========================================================================
MODULE actlite_3yr_like

! ===========================================================================
  logical :: use_act    = .true.
  logical :: use_spt    = .false.
  character(len=*), parameter,public :: ACT_data_dir = 'data/'

! ===========================================================================  
  character(len=*), parameter, public :: act_likelihood_version='ACTlite_3yr_v2p2'

  integer,public :: tt_lmax                = 3750  
  integer, parameter,public :: nbin1       = 42  !max nbins in ACT data	
  integer, parameter,public :: nbin2       = 47  !max nbins in SPT data
  integer, parameter,public :: nbin       = 89  !max nbins in SPT data   
  !-------------------------------------------------------
  real(8) :: sig_e            =  0.04d0  !AR1 fractional error in power
  real(8) :: sig_s            =  0.04d0  
  real(8), parameter    :: PI    = 3.14159265358979323846264d0
  REAL(8) ::  btt_dat(nbin),btt_sig(nbin)
  REAL(8) ::  covmat(nbin,nbin),cov_tot(nbin,nbin),bval(nbin)
  REAL(8), dimension(:,:), allocatable :: fisher,win_func_e, win_func_s,win_func_t,win_func
 REAL(8), dimension(:), allocatable :: diff_vec
  integer:: lmax_win           = 10000 !ell max of the full window functions  
  integer:: lmax_win_k         = 3300 !ell max of SPT windows
  integer:: bmax0              = 33   !number of bins in full window function
  integer, parameter :: nbin_e   = 21  !bins in equa and south vector
  integer, parameter :: b1 = 4   !First bin of full window function used

  PRIVATE
  public :: act_likelihood_init
  public :: act_likelihood_compute
  public :: get_free_lun
contains
  
  ! ===========================================================================
  SUBROUTINE act_likelihood_init
    ! ===========================================================================
    
    IMPLICIT NONE
    
    INTEGER  :: i,j,lun,il,info
    CHARACTER(LEN=240) :: ttfilename, bblefilename, bblsfilename,bbltfilename,covfilename
    LOGICAL  :: good
    
    print *, 'Initializing ACT likelihood, version '//act_likelihood_version
    
    !-----------------------------------------------
    ! set file names
    !-----------------------------------------------
    
    ttfilename  = trim(ACT_data_dir)//'ACT+SPT_cl.dat'
    covfilename = trim(ACT_data_dir)//'ACT+SPT_cov.dat'
    bblsfilename = trim(ACT_data_dir)//'Bbl_148_south_v2p2.dat'
    bblefilename = trim(ACT_data_dir)//'Bbl_148_equa_v2p2.dat'
    bbltfilename = trim(ACT_data_dir)//'Bbl_150_spt_v2p2.dat'

    !-----------------------------------------------
    ! load spectrum and band power window functions 
    !-----------------------------------------------
    
    inquire(file=ttfilename,exist = good)
    if(.not.good)then
       write(*,*) 'cant find', trim(ttfilename), trim(ACT_data_dir)
       stop
    endif
    call get_free_lun( lun )
    open(unit=lun,file=ttfilename,form='formatted',status='unknown',action='read')
    
    do i=1,nbin1 !read ACT
       read(lun,*) bval(i),btt_dat(i), btt_sig(i)
       btt_dat(i)=btt_dat(i)*(2.d0*PI)/(bval(i)*(bval(i)+1.d0)) !Convert back to Cl
       btt_sig(i)=btt_sig(i)*(2.d0*PI)/(bval(i)*(bval(i)+1.d0)) !Convert back to Cl
    enddo
    do i=nbin1+1,nbin !read SPT
       read(lun,*) bval(i),btt_dat(i), btt_sig(i)
    enddo
    close(lun)
    
    call get_free_lun( lun )
    open(unit=lun,file=covfilename,form='formatted',status='unknown',action='read')
    
    do il = 1,nbin
       read(lun,*) (covmat(i,il),i=1,nbin)
    enddo
    close(lun)

    inquire (file=bblefilename,exist = good)
    if(.not.good)then
       write(*,*) 'cant find', trim(bblefilename), trim(ACT_data_dir)
       stop
    endif
    call get_free_lun( lun )
    open(unit=lun,file=bblefilename,form='formatted',status='unknown',action='read')
    
    allocate(win_func_e(1:bmax0,1:lmax_win)) !Defined over ACT's full ell range 
    do il = 2, lmax_win
	read(lun,*) i, (win_func_e(i,il), i=1,bmax0) 
	enddo	
    close(lun)

    inquire (file=bblsfilename,exist = good)
    if(.not.good)then
       write(*,*) 'cant find', trim(bblsfilename), trim(ACT_data_dir)
       stop
    endif
    call get_free_lun( lun )
    open(unit=lun,file=bblsfilename,form='formatted',status='unknown',action='read')
    
    allocate(win_func_s(1:bmax0,1:lmax_win))
    do il = 2, lmax_win
       read(lun,*) i, (win_func_s(i,il), i=1,bmax0) 
    enddo
    close(lun)

    open(unit=lun,file=bbltfilename,form='formatted',status='unknown',action='read')

    allocate(win_func_t(1:nbin2,1:lmax_win_k))
    do il = 2, lmax_win_k
       read(lun,*) i, (win_func_t(i,il), i=1,nbin2)
    enddo
    close(lun)

!Combine the two ACT windows into 1 array - ACT-E followed by ACT-S
    allocate(win_func(1:nbin1,2:lmax_win)) 
    do i=1,nbin_e
       win_func(i,2:lmax_win)=win_func_e(i+b1-1,2:lmax_win) 
       win_func(i+nbin_e,2:lmax_win)=win_func_s(i+b1-1,2:lmax_win)
    enddo
    
  END SUBROUTINE act_likelihood_init
  
  ! ===========================================================================
  
  SUBROUTINE act_likelihood_compute(cltt,like)
    
 ! ===========================================================================
    
    IMPLICIT NONE
    REAL(8), intent(in) :: cltt(2:*)
    REAL(8), intent(out) :: like
    INTEGER :: bin_no,lun,il,i,j,info
    REAL(8) :: cltt_temp(2:lmax_win)
    REAL(8) :: btt_th(nbin),btt_dum(nbin),diffs(nbin)
    REAL(8) :: tmp_e(nbin,1),tmp_s(nbin,1)
    REAL(8) :: dlnlike,lndet
    REAL(8), dimension(:), allocatable :: tmp
    Like = 0.d0

    cltt_temp(2:lmax_win)=0.d0  !Neglect theory Cl et l>3750
    cltt_temp(2:tt_lmax)=cltt(2:tt_lmax)

!Multiply theory Cls by SPT bandpowers
    btt_th(nbin1+1:nbin)=MATMUL(win_func_t(1:nbin2,2:lmax_win_k),cltt_temp(2:lmax_win_k))

!Convert theory back to Cl
    do il=2,tt_lmax
       cltt_temp(il)=cltt_temp(il)*2.d0*PI/(dble(il)*(dble(il)+1.d0)) 
    enddo
 
!Multiply theory Cls by ACT bandpower matrix
    btt_th(1:nbin1)=MATMUL(win_func(1:nbin1,2:lmax_win),cltt_temp(2:lmax_win))

    !Basic chisq 
    diffs = btt_dat - btt_th

    !Replace cov matrix with one inflated for calibration error (could put in quick form to avoid re-inverting at every step, but not done here)
    tmp_e(:,:)=0.d0
    tmp_s(:,:)=0.d0
    tmp_e(1:nbin_e,1)=btt_th(1:nbin_e) !ACT-E theory
    tmp_s(nbin_e+1:nbin1,1)=btt_th(nbin_e+1:nbin1) !ACT-S theory
    cov_tot(:,:) = covmat(:,:)+sig_e**2*matmul(tmp_e,transpose(tmp_e))+sig_s**2*matmul(tmp_s,transpose(tmp_s))
    
    !Only ACT
    if((use_act .eqv. .true.) .and. (use_spt .eqv. .false.)) then
       bin_no=nbin1
       allocate(fisher(nbin1,nbin1))
       allocate(diff_vec(nbin1),tmp(nbin1))
       diff_vec(:)=diffs(1:nbin1)
       fisher(:,:) = cov_tot(1:nbin1,1:nbin1)
    !Only SPT
    else if((use_act .eqv. .false.) .and. (use_spt .eqv. .true.)) then
       bin_no=nbin2
       allocate(fisher(nbin2,nbin2))
       allocate(diff_vec(nbin2),tmp(nbin2))
       diff_vec(:)=diffs(nbin1+1:nbin)
       fisher(:,:) = cov_tot(nbin1+1:nbin,nbin1+1:nbin)
    !ACT and SPT - use ACT-E
    else if ((use_act .eqv. .true.) .and. (use_spt .eqv. .true.)) then
       bin_no=nbin_e+nbin2
       allocate(fisher(bin_no,bin_no))
       allocate(diff_vec(bin_no),tmp(bin_no))

       diff_vec(1:nbin_e)=diffs(1:nbin_e) !ACT-E data
       diff_vec(nbin_e+1:bin_no)=diffs(nbin1+1:nbin) !SPT data

       fisher(1:nbin_e,1:nbin_e) =cov_tot(1:nbin_e,1:nbin_e) !ACT-E block
       fisher(nbin_e+1:bin_no,nbin_e+1:bin_no) = cov_tot(nbin1+1:nbin,nbin1+1:nbin)  !SPT block
       fisher(1:nbin_e,nbin_e+1:bin_no)=cov_tot(1:nbin_e,nbin1+1:nbin) !offdiag
       fisher(nbin_e+1:bin_no,1:nbin_e)=cov_tot(nbin1+1:nbin,1:nbin_e) !offdiag
    else
       write(*,*) 'Fail: no options chosen'
    endif
    
    !Invert covmat
    call dpotrf('U',bin_no,fisher,bin_no,info)
    if(info.ne.0)then
       print*, ' info in dpotrf =', info
       stop
    endif
    call dpotri('U',bin_no,fisher,bin_no,info)
    if(info.ne.0)then
       print*, ' info in dpotri =', info
       stop
    endif
    do i=1,bin_no
       do j=i,bin_no
          fisher(j,i)=fisher(i,j)
       enddo
    enddo
 
    tmp=matmul(fisher,diff_vec)
    dlnlike=sum(tmp*diff_vec)
    deallocate(fisher,diff_vec,tmp)
    !Leave out lndet term as variation is <0.1 over calibration range considered

    Like = dlnlike/2.d0
    
  end SUBROUTINE act_likelihood_compute
  
 subroutine get_free_lun( lun )

    implicit none
    integer, intent(out) :: lun
    
    integer, save :: last_lun = 19
    logical :: used
    lun = last_lun
    do
       inquire( unit=lun, opened=used )
       if ( .not. used ) exit
       lun = lun + 1
    end do
    
    last_lun = lun
  end subroutine get_free_lun


END MODULE actlite_3yr_like

