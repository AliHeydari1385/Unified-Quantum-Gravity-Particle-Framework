This software is for the use of the ACT 3-year data, and is based on the publicly available WMAP likelihood code.

Please reference Dunkley et al 2013, astro-ph/1301.0776 when using this likelihood. If using the SPT part, please also reference Calabrese et al 2013, astro-ph/1302.1841, and note that the SPT likelihood is derived from the SPT data in Story et al 2012, astro-ph/1210.7231.

It is written in Fortran 90 and requires Cholesky matrix inversion routines; LAPACK will provide the routines.

To build it, edit the Makefile to support your environment; the provided Makefile shows a possible configuration.

Then:
make                    # Compiles the likelihood 
./test			# Runs a test file

Differences on the order of 0(0.001) are normal between computer platforms.

Modification History:
- 2013-Jan-07 - Initial release v1. Code based on public WMAP likelihood. Code written by E. Calabrese and J. Dunkley
- 2013-Feb-14 - Second release v2. Updated to use improved binning of the ACT data, and also includes CMB bandpowers estimated from the SPT Story et al data.

- 2013-Oct - Updated release v2p2. Uses refined beams for the ACT data, but effect on parameters small (0.1 sigma level). 

===============================================================================
        Tarball contents.
===============================================================================
The tarball contains a data subdirectory that contains input files, and the following primary files:

-------------------------------------------------------------------------------
0) README.txt
-------------------------------------------------------------------------------
This descriptive file.

-------------------------------------------------------------------------------
1) test.f90
-------------------------------------------------------------------------------
A wrapper program shows you how to call the likelihood code and allow you to
run and test it. 

-------------------------------------------------------------------------------
2) ACTlite_3yr_like.f90  
-------------------------------------------------------------------------------
The central likelihood routine 
- Has two flags 'use_spt' and 'use_act' to choose which data to use
- Using ACT alone, computes a chisq for the 3-year ACT CMB-only data, summed over the ACT-E and ACT-S data (21 bins each).
- Using SPT alone, computes a chisq for the SPT CMB-only data, derived from Story et al, summed over 47 bins.
- If using ACT and SPT together, computes a chisq for ACT-E and SPT, summed over 68 bins

- Analytically marginalizes over two calibration factors for ACT
- No extra nuisance parameters are required
- The CMB theory spectrum is required up to lmax=3750
