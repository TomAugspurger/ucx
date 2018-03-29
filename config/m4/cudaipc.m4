#
# Check for CUDA-IPC support
#
cudaipc_happy="no"
AC_ARG_WITH([cudaipc],
           [AS_HELP_STRING([--with-cudaipc=(DIR)], [Enable the use of CUDA IPC (default is guess).])],
           [], [with_cudaipc=guess])
AS_IF([test "x$with_cudaipc" != xno],
      [
       save_CPPFLAGS="$CPPFLAGS"
       save_CFLAGS="$CFLAGS"
       save_LDFLAGS="$LDFLAGS"
       AS_IF([test ! -z "$with_cudaipc" -a "x$with_cudaipc" != "xyes" -a "x$with_cudaipc" != "xguess"],
             [
                ucx_check_cudaipc_dir="$with_cudaipc"
                AS_IF([test -d "$with_cudaipc/lib64"],[libsuff="64"],[libsuff=""])
                ucx_check_cudaipc_libdir="$with_cudaipc/lib$libsuff"
                CPPFLAGS="-I$with_cudaipc/include $save_CPPFLAGS"
                LDFLAGS="-L$ucx_check_cudaipc_libdir $save_LDFLAGS"
             ])
       AS_IF([test ! -z "$with_cudaipc_libdir" -a "x$with_cudaipc_libdir" != "xyes"],
             [
	      ucx_check_cudaipc_libdir="$with_cudaipc_libdir"
              LDFLAGS="-L$ucx_check_cudaipc_libdir $save_LDFLAGS"
	     ])
       AC_CHECK_HEADERS([cuda.h],
                        [AC_CHECK_LIB([cuda] , [cuPointerGetAttribute],
                         [
			  cudaipc_happy="yes"
			  transports="${transports},cudaipc"
			 ],
                         [AC_MSG_WARN([CUDA runtime not detected. Disable.])
                          cudaipc_happy="no"
			 ])
			],
			[cudaipc_happy="no"])
      ],
      [AC_MSG_WARN([CUDAIPC was explicitly disabled])
       AC_DEFINE([HAVE_CUDAIPC], [0], [Disable the use of CUDA-IPC])]
)

AM_CONDITIONAL([HAVE_CUDAIPC], [test "x$cudaipc_happy" != xno])
				     
