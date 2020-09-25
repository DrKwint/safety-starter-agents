(defdfa int_counter_with_null_reset ((-1 0 1 R N) (-3 -2 -1 0 1 2 3 4) 0 (-3 3))
  ((4 1 1) (0 1 1) (1 2 1) (2 3 1) ;go right on a 1
   (1 0 R -1) (2 0 R -1) (3 0 R -1) ;reset to 0 from right on a R or -1
   (4 -1 -1) (0 -1 -1) (-1 -2 -1) (-2 -3 -1) ;go left on a -1
   (-1 0 R 1) (-2 0 R 1) (-3 0 R 1) ;reset to 0 from left on a R or 1
   (4 0 0) (-3 -3 0) (-2 -2 0) (-1 -1 0) (0 0 0) (1 1 0) (2 2 0) (3 3 0) ; go from NULL to zero or stay put on a zero
   (-3 4 N) (-2 4 N) (-1 4 N) (0 4 N) (1 4 N) (2 4 N) (3 4 N) (4 4 N)) ; goto NULL on a null from anywhere
  ("A counter which will count in positive or negative direction by 1 unil reset to 0 by another token or goes to NULL (denoted with 4) with 4 token"))
