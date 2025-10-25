OBJS =  ACTlite_3yr_like.o 

RM = rm -f

# Compiler specific commands

#ifort 
F90     = mpif90
FFLAGS  = -openmp -O3 
LIBS =  -mkl

#------------------


#                       Rules.

all: test

test: $(OBJS) test.o
	$(F90) $(FFLAGS) -o $@ test.o $(OBJS) $(LIBS)

%.o: %.f90
	$(F90) $(FFLAGS) $(INCS) -c -o $@ $<

clean:
	$(RM) *.o *.mod *.log *~ *.a test
distclean: clean
	$(RM) test


